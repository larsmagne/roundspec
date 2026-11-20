/*
 * roundspec - a round, diagonal, stereo spectrum display for ALSA + X11.
 *
 * The window is a circle, divided (invisibly) by an axis that starts as a
 * 45 degree diagonal from bottom-left to top-right and slowly turns.  Every frequency band is a spoke starting at the centre; at
 * full level it reaches the rim.  The left channel's spokes fan out over
 * the upper-left half, the right channel's over the lower-right half.
 * Bass is next to the top-right end of the diagonal, treble next to the
 * bottom-left end, mirrored on both sides.  Spoke colour glides from blue
 * (bass) over purple to red (treble).
 *
 * Build:  cc -O2 -o roundspec roundspec.c -lasound -lX11 -lXext -lm
 *         (Debian: apt install libasound2-dev libx11-dev libxext-dev)
 *
 * Usage:  roundspec [-w width] [-s scale] [-c colours] [-r secs]
 *                   [-g geometry] [-T] [capture-device] [rate]
 *         roundspec -T ...       (do not keep the window on top)
 *         roundspec -d ...       (print what is being captured, once a second)
 *         roundspec -u ...       (unmirrored: both channels run the same way
 *                                 round the circle, so the right channel's
 *                                 bass follows the left channel's treble)
 *         roundspec -f ...       (flip: swap the bass and treble ends)
 *         roundspec -n 1024 ...  (FFT size, a power of two from 256 to 16384;
 *                                 default 2048.  Smaller reacts faster,
 *                                 larger separates bass notes better.)
 *         roundspec -g 800 ...            (800x800 window)
 *         roundspec -g 800x800+100+50 ... (also place it at 100,50;
 *                                          -0-0 is the bottom-right corner)
 *         roundspec -r 20 ...    (axis makes a full turn every 20 seconds;
 *                                 default 60, negative turns the other way,
 *                                 0 keeps it fixed at 45 degrees)
 *         roundspec -c green,yellow,red plughw:Loopback,1,0
 *                   (bass-to-treble gradient: 2 or more X colour names or
 *                    '#rrggbb' values, separated by commas)
 *         roundspec -w 6 plughw:Loopback,1,0     (6 pixel wide spokes)
 *         roundspec -s 1.5 plughw:Loopback,1,0   (spokes 1.5 times longer)
 *         roundspec plughw:Loopback,1,0
 *         roundspec -            (raw S16 stereo on stdin, e.g. an mpd fifo)
 *
 * The window is round, has no title bar (if the window manager agrees) and
 * keeps itself on top of other windows: drag it with the left mouse
 * button to move it; q or Escape quits.
 */

#include <alsa/asoundlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/Xatom.h>
#include <X11/extensions/shape.h>
#include <math.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NMAX    16384       /* largest FFT size allowed with -n     */
#define PERIOD  1024        /* most frames fetched in one read      */
#define FPS     60          /* screen updates per second, at most   */
#define BANDS   64          /* spikes per channel                   */
#define F_LO    40.0        /* lowest frequency shown, Hz           */
#define F_HI    16000.0     /* highest frequency shown, Hz          */
#define DB_FLOOR 70.0       /* dynamic range shown, dB below full   */
#define DECAY_T 0.18        /* seconds for a spoke to fall to 1/e   */
#define COLOUR  "#b83030"   /* the circle outline                   */

/* Spoke colours glide from bass to treble through this list; set with -c. */
#define MAX_STOPS 8
static const char *gradient = "#4060ff,#b030d0,#ff3030";   /* blue, purple, red */

static int   N = 2048;              /* FFT size (power of two); set with -n */
static float ring[2][NMAX];         /* latest N samples per channel */
static float decay = 1;             /* fall-back factor for this frame */
static int   ringpos;
static float window[NMAX];
static float level[2][BANDS];       /* smoothed 0..1 band levels    */
static int   band_lo[BANDS], band_hi[BANDS];
static unsigned long band_colour[BANDS];
static int   spoke_width = 2;       /* pixels; set with -w          */
static int   unmirrored = 0;        /* set with -u                  */
static int   flipped = 0;           /* set with -f                  */
static double turn_secs = 60.0;     /* seconds per full turn; set with -r */
static double spoke_scale = 1.0;    /* length multiplier; set with -s */

/* In-place iterative radix-2 FFT. */
static void fft(float *re, float *im, int n)
{
    int i, j, len;

    for (i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / len;
        float wr = cos(ang), wi = sin(ang);
        for (i = 0; i < n; i += len) {
            float cr = 1, ci = 0;
            for (j = 0; j < len / 2; j++) {
                int a = i + j, b = a + len / 2;
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
                float t = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = t;
            }
        }
    }
}

static void setup_tables(double rate)
{
    int i;

    for (i = 0; i < N; i++)                      /* Hann window */
        window[i] = 0.5 - 0.5 * cos(2.0 * M_PI * i / (N - 1));

    for (i = 0; i < BANDS; i++) {                /* log-spaced bands */
        double f0 = F_LO * pow(F_HI / F_LO, (double)i / BANDS);
        double f1 = F_LO * pow(F_HI / F_LO, (double)(i + 1) / BANDS);
        int lo = (int)(f0 * N / rate), hi = (int)(f1 * N / rate);
        if (lo < 1) lo = 1;
        if (hi < lo) hi = lo;
        if (hi > N / 2 - 1) hi = N / 2 - 1;
        if (lo > hi) lo = hi;
        band_lo[i] = lo;
        band_hi[i] = hi;
    }
}

static void analyse(int ch)
{
    static float re[NMAX], im[NMAX];
    int i, b;

    for (i = 0; i < N; i++) {
        re[i] = ring[ch][(ringpos + i) % N] * window[i];
        im[i] = 0;
    }
    fft(re, im, N);

    for (b = 0; b < BANDS; b++) {
        float peak = 0, v;
        for (i = band_lo[b]; i <= band_hi[b]; i++) {
            float m = re[i] * re[i] + im[i] * im[i];
            if (m > peak) peak = m;
        }
        /* N/4 = N/2 (one-sided) * 0.5 (Hann gain); samples are -1..1 */
        v = 10.0 * log10(peak / ((N / 4.0) * (N / 4.0)) + 1e-12);
        v = (v + DB_FLOOR) / DB_FLOOR;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        level[ch][b] = v > level[ch][b] * decay ? v : level[ch][b] * decay;
    }
}

static void draw(Display *dpy, Drawable d, GC gc, unsigned long fg,
                 unsigned long bg, int w, int h)
{
    double cx = w / 2.0, cy = h / 2.0;
    double R = (w < h ? w : h) / 2.0 - 2;
    double axis = M_PI / 4;             /* direction of the dividing axis */
    struct timespec now;

    if (turn_secs != 0) {               /* let the axis turn slowly */
        static double t0;
        double t;
        clock_gettime(CLOCK_MONOTONIC, &now);
        t = now.tv_sec + now.tv_nsec / 1e9;
        if (t0 == 0)
            t0 = t;                     /* start at 45 degrees */
        axis += 2 * M_PI * fmod(t - t0, fabs(turn_secs)) / turn_secs;
    }
    int ch, i;

    XSetForeground(dpy, gc, bg);
    XFillRectangle(dpy, d, gc, 0, 0, w, h);
    if (R < 10)
        return;
    XSetForeground(dpy, gc, fg);

    XSetLineAttributes(dpy, gc, 2, LineSolid, CapRound, JoinRound);
    XDrawArc(dpy, d, gc, cx - R, cy - R, 2 * R, 2 * R, 0, 360 * 64);

    /*
     * One spoke per band, from the centre towards the rim; a band at full
     * level reaches the circle.  The left channel fans anticlockwise over
     * the half circle starting at the axis direction, the right channel
     * mirrors it clockwise, so bass sits next to one end of the axis and
     * treble next to the other end on both sides.  With the axis at its
     * starting 45 degrees, left is the upper-left half and bass is top-right.
     */
    XSetLineAttributes(dpy, gc, spoke_width, LineSolid, CapRound, JoinRound);
    for (ch = 0; ch < 2; ch++) {
        double dir = ch == 0 ? 1.0 : -1.0;
        for (i = 0; i < BANDS; i++) {
            double frac = (i + 0.5) / BANDS;    /* 0 = bass, 1 = treble */
            double a;
            if (flipped)
                frac = 1 - frac;
            if (unmirrored)     /* right half carries on anticlockwise */
                a = axis + (ch == 0 ? 0 : M_PI) + M_PI * frac;
            else                /* right half is a mirror image of the left */
                a = axis + dir * M_PI * frac;
            double v = level[ch][i] * spoke_scale;
            double r = (v > 1 ? 1 : v) * R;     /* never past the rim */
            XSetForeground(dpy, gc, band_colour[i]);
            XDrawLine(dpy, d, gc, cx, cy, cx + r * cos(a), cy - r * sin(a));
        }
    }
}

/*
 * Fetch up to PERIOD stereo frames without waiting.  Returns the number
 * of frames read, 0 if nothing has arrived yet, or -1 at end of input.
 */
static int read_audio(snd_pcm_t *pcm, short *buf)
{
    if (pcm) {
        /*
         * Capture as 32-bit: snd-aloop makes both ends of the loopback use
         * the same sample format, and the playback end is S32 when it is
         * tied to a card like the Hammerfall.  Asking for S16 here would
         * lock the loopback to S16 and make the player's open fail.
         */
        static int32_t buf32[PERIOD * 2];
        snd_pcm_sframes_t n;
        int i;

        /* A capture stream does not start by itself on a small read. */
        if (snd_pcm_state(pcm) == SND_PCM_STATE_PREPARED) {
            int err = snd_pcm_start(pcm);
            static time_t last_msg;
            if (err < 0 && time(NULL) != last_msg) {
                last_msg = time(NULL);
                fprintf(stderr, "cannot start capture: %s\n", snd_strerror(err));
            }
        }

        /*
         * Ask how much is waiting and read no more than that, so the read
         * can never block (and never keeps the main loop from drawing).
         */
        n = snd_pcm_avail(pcm);
        if (n == 0)
            return 0;
        if (n > 0)
            n = snd_pcm_readi(pcm, buf32, n < PERIOD ? n : PERIOD);
        if (n == -EAGAIN)
            return 0;
        if (n < 0) {
            /* overrun, or the player reopened the other end: restart */
            static time_t last_msg;
            if (time(NULL) != last_msg) {       /* at most one line a second */
                last_msg = time(NULL);
                fprintf(stderr, "capture: %s; restarting\n", snd_strerror(n));
            }
            if (snd_pcm_recover(pcm, n, 1) < 0 && snd_pcm_prepare(pcm) < 0) {
                fprintf(stderr, "cannot restart capture: %s\n", snd_strerror(n));
                return -1;
            }
            return 0;
        }
        for (i = 0; i < n * 2; i++)
            buf[i] = buf32[i] >> 16;
        return n;
    } else {
        /* Take whatever has arrived; keep a partial frame for next time. */
        static char pend[PERIOD * 2 * sizeof(short)];
        static size_t have;
        struct pollfd pfd = { 0, POLLIN, 0 };
        size_t frames;
        ssize_t n;

        if (poll(&pfd, 1, 0) <= 0)
            return 0;
        n = read(0, pend + have, sizeof pend - have);
        if (n <= 0)
            return -1;
        have += n;
        frames = have / (2 * sizeof(short));
        memcpy(buf, pend, frames * 2 * sizeof(short));
        have -= frames * 2 * sizeof(short);
        memmove(pend, pend + frames * 2 * sizeof(short), have);
        return frames;
    }
}

/* Clip the window to the circle, so only the disc is visible on screen. */
static void set_round_shape(Display *dpy, Window win, int w, int h)
{
    int d = w < h ? w : h;
    Pixmap mask = XCreatePixmap(dpy, win, w, h, 1);
    GC mgc = XCreateGC(dpy, mask, 0, NULL);

    XSetForeground(dpy, mgc, 0);
    XFillRectangle(dpy, mask, mgc, 0, 0, w, h);
    XSetForeground(dpy, mgc, 1);
    XFillArc(dpy, mask, mgc, (w - d) / 2, (h - d) / 2, d, d, 0, 360 * 64);
    XShapeCombineMask(dpy, win, ShapeBounding, 0, 0, mask, ShapeSet);
    XFreeGC(dpy, mgc);
    XFreePixmap(dpy, mask);
}

int main(int argc, char **argv)
{
    int opt;
    const char *geometry = NULL;
    int on_top = 1, obscured = 0, debug = 0;
    double last_raise = 0;

    while ((opt = getopt(argc, argv, "w:s:c:r:g:Tn:duf")) != -1) {
        if (opt == 'w' && atoi(optarg) > 0) {
            spoke_width = atoi(optarg);
        } else if (opt == 's' && atof(optarg) > 0) {
            spoke_scale = atof(optarg);
        } else if (opt == 'r') {
            turn_secs = atof(optarg);
        } else if (opt == 'n' && atoi(optarg) >= 256 && atoi(optarg) <= NMAX &&
                   (atoi(optarg) & (atoi(optarg) - 1)) == 0) {
            N = atoi(optarg);
        } else if (opt == 'u') {
            unmirrored = 1;
        } else if (opt == 'f') {
            flipped = 1;
        } else if (opt == 'd') {
            debug = 1;
        } else if (opt == 'T') {
            on_top = 0;
        } else if (opt == 'g') {
            geometry = optarg;
        } else if (opt == 'c') {
            gradient = optarg;
        } else {
            fprintf(stderr, "usage: %s [-w width] [-s scale] [-c from,to] [-r secs] "
                    "[-g WxH+X+Y] [-T] [-n fftsize] [-u] [-f] [-d] "
                    "[capture-device] [rate]\n",
                    argv[0]);
            return 1;
        }
    }
    const char *dev = optind < argc ? argv[optind] : "default";
    unsigned rate = optind + 1 < argc ? atoi(argv[optind + 1]) : 44100;
    snd_pcm_t *pcm = NULL;
    short buf[PERIOD * 2];
    int err, w = 500, h = 500, running = 1;

    if (strcmp(dev, "-") != 0) {
        if ((err = snd_pcm_open(&pcm, dev, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
            fprintf(stderr, "cannot open %s: %s\n", dev, snd_strerror(err));
            return 1;
        }
        if ((err = snd_pcm_set_params(pcm, SND_PCM_FORMAT_S32,
                                      SND_PCM_ACCESS_RW_INTERLEAVED,
                                      2, rate, 1, 100000)) < 0) {
            fprintf(stderr, "cannot configure %s: %s\n", dev, snd_strerror(err));
            return 1;
        }
        snd_pcm_nonblock(pcm, 1);
    }
    if (debug) {
        snd_pcm_uframes_t bs = 0, ps = 0;
        if (pcm)
            snd_pcm_get_params(pcm, &bs, &ps);
        fprintf(stderr, "device %s, rate %u, fft %d, buffer %lu, period %lu\n",
                dev, rate, N, (unsigned long)bs, (unsigned long)ps);
    }
    setup_tables(rate);

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "cannot open X display\n");
        return 1;
    }
    int scr = DefaultScreen(dpy);
    unsigned long bg = BlackPixel(dpy, scr), fg = WhitePixel(dpy, scr);
    XColor col;
    if (XParseColor(dpy, DefaultColormap(dpy, scr), COLOUR, &col) &&
        XAllocColor(dpy, DefaultColormap(dpy, scr), &col))
        fg = col.pixel;

    /* Parse the comma-separated gradient and give every band its colour. */
    XColor stop[MAX_STOPS];
    int nstops = 0;
    char *list = strdup(gradient), *name;
    for (name = strtok(list, ","); name; name = strtok(NULL, ",")) {
        if (nstops == MAX_STOPS ||
            !XParseColor(dpy, DefaultColormap(dpy, scr), name, &stop[nstops])) {
            fprintf(stderr, "bad colour list \"%s\" (at \"%s\"); use up to %d "
                    "names or #rrggbb values separated by commas\n",
                    gradient, name, MAX_STOPS);
            return 1;
        }
        nstops++;
    }
    free(list);
    for (int i = 0; i < BANDS; i++) {
        double t = nstops > 1 ? (double)i / (BANDS - 1) * (nstops - 1) : 0;
        int k = (int)t;
        if (k > nstops - 2)
            k = nstops > 1 ? nstops - 2 : 0;
        double f = t - k;
        XColor *p = &stop[k], *q = &stop[nstops > 1 ? k + 1 : k], c;
        c.red   = p->red   + f * (q->red   - p->red);
        c.green = p->green + f * (q->green - p->green);
        c.blue  = p->blue  + f * (q->blue  - p->blue);
        c.flags = DoRed | DoGreen | DoBlue;
        band_colour[i] = XAllocColor(dpy, DefaultColormap(dpy, scr), &c)
                         ? c.pixel : fg;
    }

    int wx = 0, wy = 0, placed = 0;
    if (geometry) {
        unsigned gw, gh;
        int f = XParseGeometry(geometry, &wx, &wy, &gw, &gh);
        if (f & WidthValue)
            w = h = gw;                 /* "-g 800" means 800x800 */
        if (f & HeightValue)
            h = gh;
        if (f & XNegative)
            wx += DisplayWidth(dpy, scr) - w;
        if (f & YNegative)
            wy += DisplayHeight(dpy, scr) - h;
        placed = f & (XValue | YValue);
        if (!(f & (WidthValue | XValue | YValue)) || w < 20 || h < 20) {
            fprintf(stderr, "bad geometry \"%s\"; use e.g. 800, 800x600 or "
                    "800x800+100+50\n", geometry);
            return 1;
        }
    }

    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), wx, wy, w, h,
                                     0, bg, bg);
    XStoreName(dpy, win, "roundspec");
    XSizeHints sh;                      /* tell the window manager we mean it */
    sh.flags = USSize | (placed ? USPosition : 0);
    sh.x = wx; sh.y = wy; sh.width = w; sh.height = h;
    XSetWMNormalHints(dpy, win, &sh);
    XSelectInput(dpy, win, StructureNotifyMask | KeyPressMask |
                 ButtonPressMask | Button1MotionMask | VisibilityChangeMask);

    /*
     * Staying on top.  Modern window managers do it for us when asked with
     * _NET_WM_STATE_ABOVE.  Old ones such as twm know nothing about that,
     * so we also watch for being covered (VisibilityNotify) and raise
     * ourselves in the main loop.
     */
    if (on_top) {
        Atom state = XInternAtom(dpy, "_NET_WM_STATE", False);
        Atom above = XInternAtom(dpy, "_NET_WM_STATE_ABOVE", False);
        XChangeProperty(dpy, win, state, XA_ATOM, 32, PropModeReplace,
                        (unsigned char *)&above, 1);
    }

    /* Ask the window manager for no title bar or border. */
    Atom motif = XInternAtom(dpy, "_MOTIF_WM_HINTS", False);
    long hints[5] = { 2, 0, 0, 0, 0 };          /* flags=decorations, none */
    XChangeProperty(dpy, win, motif, motif, 32, PropModeReplace,
                    (unsigned char *)hints, 5);
    set_round_shape(dpy, win, w, h);
    int drag_x = 0, drag_y = 0;                 /* pointer offset while dragging */
    Atom wm_delete = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wm_delete, 1);
    XMapWindow(dpy, win);
    XFlush(dpy);

    GC gc = XCreateGC(dpy, win, 0, NULL);
    XSetLineAttributes(dpy, gc, 2, LineSolid, CapRound, JoinRound);
    XSetGraphicsExposures(dpy, gc, False);
    int depth = DefaultDepth(dpy, scr);
    Pixmap pix = XCreatePixmap(dpy, win, w, h, depth);

    double last_draw = 0, last_audio = 0, dbg_time = 0;
    long dbg_frames = 0, dbg_reads = 0, dbg_draws = 0;
    int dbg_peak = 0;
    while (running) {
        /*
         * Take everything that has arrived, so the picture always shows the
         * newest sound and never falls behind.  The capture buffer is big
         * (so it cannot overflow) but adds no delay, because it is emptied
         * every couple of milliseconds.
         */
        int i, n, total = 0;

        int reads = 0;

        while ((n = read_audio(pcm, buf)) > 0) {
            for (i = 0; i < n * 2; i++)
                if (abs(buf[i]) > dbg_peak)
                    dbg_peak = abs(buf[i]);
            dbg_frames += n;
            dbg_reads++;
            for (i = 0; i < n; i++) {
                ring[0][ringpos] = buf[2 * i] / 32768.0f;
                ring[1][ringpos] = buf[2 * i + 1] / 32768.0f;
                ringpos = (ringpos + 1) % N;
            }
            total += n;
            if (++reads == 64)          /* never let reading starve drawing */
                break;
        }
        if (n < 0)
            break;

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        double tnow = ts.tv_sec + ts.tv_nsec / 1e9;
        if (total)
            last_audio = tnow;
        if (debug && tnow - dbg_time >= 1) {
            float top = 0;
            for (i = 0; i < BANDS; i++)
                if (level[0][i] > top)
                    top = level[0][i];
            fprintf(stderr, "frames %ld in %ld reads, peak sample %d of 32767, "
                    "%ld pictures, longest left spoke %.2f, state %s\n",
                    dbg_frames, dbg_reads, dbg_peak, dbg_draws, top,
                    pcm ? snd_pcm_state_name(snd_pcm_state(pcm)) : "stdin");
            dbg_frames = dbg_reads = dbg_draws = dbg_peak = 0;
            dbg_time = tnow;
        }
        if (tnow - last_draw < 1.0 / FPS) {
            usleep(2000);               /* too soon for another picture */
            continue;
        }
        if (tnow - last_audio > 0.2)    /* no audio: let the spokes fall */
            memset(ring, 0, sizeof ring);
        decay = last_draw ? exp(-(tnow - last_draw) / DECAY_T) : 1;
        last_draw = tnow;

        while (XPending(dpy)) {
            XEvent ev;
            XNextEvent(dpy, &ev);
            if (ev.type == ConfigureNotify &&
                (ev.xconfigure.width != w || ev.xconfigure.height != h)) {
                w = ev.xconfigure.width;
                h = ev.xconfigure.height;
                XFreePixmap(dpy, pix);
                pix = XCreatePixmap(dpy, win, w, h, depth);
                set_round_shape(dpy, win, w, h);
            } else if (ev.type == VisibilityNotify) {
                obscured = ev.xvisibility.state != VisibilityUnobscured;
            } else if (ev.type == ButtonPress && ev.xbutton.button == Button1) {
                drag_x = ev.xbutton.x;
                drag_y = ev.xbutton.y;
            } else if (ev.type == MotionNotify) {
                XMoveWindow(dpy, win, ev.xmotion.x_root - drag_x,
                            ev.xmotion.y_root - drag_y);
            } else if (ev.type == KeyPress) {
                KeySym k = XLookupKeysym(&ev.xkey, 0);
                if (k == XK_q || k == XK_Escape)
                    running = 0;
            } else if (ev.type == ClientMessage &&
                       (Atom)ev.xclient.data.l[0] == wm_delete) {
                running = 0;
            }
        }

        if (on_top && obscured) {
            /* At most twice a second, so two such windows cannot fight hard. */
            double t = tnow;
            if (t - last_raise > 0.5) {
                XRaiseWindow(dpy, win);
                last_raise = t;
            }
        }

        analyse(0);
        analyse(1);
        dbg_draws++;
        draw(dpy, pix, gc, fg, bg, w, h);
        XCopyArea(dpy, pix, win, gc, 0, 0, w, h, 0, 0);
        XSync(dpy, False);      /* wait for the server, so frames never queue up */
    }

    XCloseDisplay(dpy);
    if (pcm)
        snd_pcm_close(pcm);
    return 0;
}

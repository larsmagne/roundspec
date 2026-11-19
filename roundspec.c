/*
 * roundspec - a round, diagonal, stereo spectrum display for ALSA + X11.
 *
 * The window is a circle, divided (invisibly) by a 45 degree diagonal from
 * bottom-left to top-right.  Every frequency band is a spoke starting at the centre; at
 * full level it reaches the rim.  The left channel's spokes fan out over
 * the upper-left half, the right channel's over the lower-right half.
 * Bass is next to the top-right end of the diagonal, treble next to the
 * bottom-left end, mirrored on both sides.
 *
 * Build:  cc -O2 -o roundspec roundspec.c -lasound -lX11 -lXext -lm
 *         (Debian: apt install libasound2-dev libx11-dev libxext-dev)
 *
 * Usage:  roundspec [-w width] [-s scale] [capture-device] [rate]
 *         roundspec -w 6 plughw:Loopback,1,0     (6 pixel wide spokes)
 *         roundspec -s 1.5 plughw:Loopback,1,0   (spokes 1.5 times longer)
 *         roundspec plughw:Loopback,1,0
 *         roundspec -            (raw S16 stereo on stdin, e.g. an mpd fifo)
 *
 * The window is round and has no title bar: drag it with the left mouse
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
#include <unistd.h>

#define N       4096        /* FFT size (power of two)              */
#define PERIOD  1024        /* frames read per screen update        */
#define BANDS   64          /* spikes per channel                   */
#define F_LO    40.0        /* lowest frequency shown, Hz           */
#define F_HI    16000.0     /* highest frequency shown, Hz          */
#define DB_FLOOR 70.0       /* dynamic range shown, dB below full   */
#define DECAY   0.88        /* how slowly spikes fall back (0..1)   */
#define COLOUR  "#b83030"

static float ring[2][N];            /* latest N samples per channel */
static int   ringpos;
static float window[N];
static float level[2][BANDS];       /* smoothed 0..1 band levels    */
static int   band_lo[BANDS], band_hi[BANDS];
static int   spoke_width = 2;       /* pixels; set with -w          */
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
    static float re[N], im[N];
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
        level[ch][b] = v > level[ch][b] * DECAY ? v : level[ch][b] * DECAY;
    }
}

static void draw(Display *dpy, Drawable d, GC gc, unsigned long fg,
                 unsigned long bg, int w, int h)
{
    double cx = w / 2.0, cy = h / 2.0;
    double R = (w < h ? w : h) / 2.0 - 2;
    XSegment seg[BANDS];
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
     * level reaches the circle.  The left channel fans over the upper-left
     * half (45 -> 225 degrees, anticlockwise), the right channel mirrors it
     * over the lower-right half, so bass sits next to the top-right end of
     * the diagonal and treble next to the bottom-left end on both sides.
     */
    XSetLineAttributes(dpy, gc, spoke_width, LineSolid, CapRound, JoinRound);
    for (ch = 0; ch < 2; ch++) {
        double dir = ch == 0 ? 1.0 : -1.0;
        for (i = 0; i < BANDS; i++) {
            double a = M_PI / 4 + dir * M_PI * (i + 0.5) / BANDS;
            double v = level[ch][i] * spoke_scale;
            double r = (v > 1 ? 1 : v) * R;     /* never past the rim */
            seg[i].x1 = cx;
            seg[i].y1 = cy;
            seg[i].x2 = cx + r * cos(a);
            seg[i].y2 = cy - r * sin(a);
        }
        XDrawSegments(dpy, d, gc, seg, BANDS);
    }
}

/*
 * Wait up to 50 ms for audio and fetch at most PERIOD stereo frames.
 * Returns the number of frames read (0 if nothing arrived, so the window
 * stays alive during silence) or -1 at end of input.
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
        if (snd_pcm_state(pcm) == SND_PCM_STATE_PREPARED)
            snd_pcm_start(pcm);

        n = snd_pcm_wait(pcm, 50);
        if (n == 0)
            return 0;
        if (n > 0)
            n = snd_pcm_readi(pcm, buf32, PERIOD);
        if (n == -EAGAIN)
            return 0;
        if (n < 0) {
            /* overrun, or the player reopened the other end: restart */
            if (snd_pcm_recover(pcm, n, 1) < 0 && snd_pcm_prepare(pcm) < 0) {
                fprintf(stderr, "read: %s\n", snd_strerror(n));
                return -1;
            }
            return 0;
        }
        for (i = 0; i < n * 2; i++)
            buf[i] = buf32[i] >> 16;
        return n;
    } else {
        struct pollfd pfd = { 0, POLLIN, 0 };
        size_t want = PERIOD * 2 * sizeof(short), got = 0;

        if (poll(&pfd, 1, 50) <= 0)
            return 0;
        while (got < want) {
            ssize_t n = read(0, (char *)buf + got, want - got);
            if (n <= 0)
                return -1;
            got += n;
        }
        return PERIOD;
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

    while ((opt = getopt(argc, argv, "w:s:")) != -1) {
        if (opt == 'w' && atoi(optarg) > 0) {
            spoke_width = atoi(optarg);
        } else if (opt == 's' && atof(optarg) > 0) {
            spoke_scale = atof(optarg);
        } else {
            fprintf(stderr,
                    "usage: %s [-w width] [-s scale] [capture-device] [rate]\n",
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
                                      2, rate, 1, 60000)) < 0) {
            fprintf(stderr, "cannot configure %s: %s\n", dev, snd_strerror(err));
            return 1;
        }
        snd_pcm_nonblock(pcm, 1);
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

    Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), 0, 0, w, h,
                                     0, bg, bg);
    XStoreName(dpy, win, "roundspec");
    XSelectInput(dpy, win, StructureNotifyMask | KeyPressMask |
                 ButtonPressMask | Button1MotionMask);

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

    while (running) {
        int i, n = read_audio(pcm, buf);

        if (n < 0)
            break;
        for (i = 0; i < n; i++) {
            ring[0][ringpos] = buf[2 * i] / 32768.0f;
            ring[1][ringpos] = buf[2 * i + 1] / 32768.0f;
            ringpos = (ringpos + 1) % N;
        }
        if (n == 0) {                    /* silence: let the spokes fall */
            for (i = 0; i < PERIOD; i++) {
                ring[0][ringpos] = ring[1][ringpos] = 0;
                ringpos = (ringpos + 1) % N;
            }
        }

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

        analyse(0);
        analyse(1);
        draw(dpy, pix, gc, fg, bg, w, h);
        XCopyArea(dpy, pix, win, gc, 0, 0, w, h, 0, 0);
        XFlush(dpy);
    }

    XCloseDisplay(dpy);
    if (pcm)
        snd_pcm_close(pcm);
    return 0;
}

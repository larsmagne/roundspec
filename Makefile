all:
	cc -O2 -o roundspec roundspec.c -lasound -lX11 -lXext -lm

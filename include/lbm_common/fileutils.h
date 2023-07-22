#include <unistd.h>   // access

static bool fileExists(const char* fname)
{
//	FILE *fp = fopen(fname, "r");
//	if (!fp) return false;
//	fclose(fp);
//	return true;
	// POSIX way is faster
	return access(fname, F_OK) == 0;
}

#include <string.h>
#include <linux/limits.h>   // PATH_MAX
#include <sys/stat.h>   // mkdir(2)
#include <errno.h>

// adapted from http://stackoverflow.com/a/2336245/119527
static int mkdir_p(const char *path, mode_t mode)
{
	const size_t len = strlen(path);
	char _path[PATH_MAX];
	char *p;

	errno = 0;

	// copy string so it is mutable
	if (len > sizeof(_path)-1) {
		errno = ENAMETOOLONG;
		return -1;
	}
	strcpy(_path, path);

	// iterate the string
	for (p = _path + 1; *p; p++) {
		if (*p == '/') {
			// temporarily truncate
			*p = '\0';

			if (mkdir(_path, mode) != 0) {
				if (errno != EEXIST)
					return -1;
			}

			*p = '/';
		}
	}

	if (mkdir(_path, S_IRWXU) != 0) {
		if (errno != EEXIST)
			return -1;
	}

	return 0;
}

#include <libgen.h>   // dirname, basename

// create parent directories and then the file
static int create_file(const char* fname)
{
	// return early if the file already exists
	if (fileExists(fname))
		return 0;

	char buffer[PATH_MAX];
	strcpy(buffer, fname);
	char* dir = dirname(buffer);

	// make sure that the parent directory exists
	mkdir_p(dir, 0777);

	// create the file
	FILE* fp = fopen(fname, "wb");
	if (fp == NULL)
	{
		fprintf(stderr, "error: failed to create file %s: %s\n", fname, strerror(errno));
		return -1;
	}
	fclose(fp);

	return 0;
}

#include <stdio.h>         // fopen, fclose, fread, fwrite, BUFSIZ
#include <sys/sendfile.h>  // sendfile
#include <fcntl.h>         // open
#include <unistd.h>        // close
#include <sys/stat.h>      // fstat
#include <sys/types.h>     // fstat

// manual copy data and meta data
static int move_file(const char* src, const char* dst)
{
	// standard C way
	/*
	FILE* in = fopen(src, "rb");
	FILE* out = fopen(dst, "wb");
	if (in==0 || out==0)
	{
		if (in==0) log("unable to open file %s", in);
		if (out==0) log("unable to create file %s", out);
		return;
	}
	size_t buffer_len = 128*1024;
	char buffer[buffer_len];
	size_t len;
	while( (len = fread(buffer, sizeof(char), buffer_len, in)) > 0 )
		fwrite(buffer, sizeof(char), len, out);
	fclose(in);
	fclose(out);
	*/

	// Linux way: https://stackoverflow.com/q/10195343  https://stackoverflow.com/a/22374134
	int source = open(src, O_RDONLY, 0);
	struct stat stat_source;
	fstat(source, &stat_source);
	int dest = open(dst, O_WRONLY | O_CREAT /*| O_TRUNC*/, stat_source.st_mode);
	off_t offset = 0LL;
	ssize_t rc = 0;
	while (offset < stat_source.st_size) {
		ssize_t count;
		off_t remaining = stat_source.st_size - offset;
		if (remaining > SSIZE_MAX)
			count = SSIZE_MAX;
		else
			count = remaining;
		rc = sendfile(dest, source, &offset, count);
		if (rc == 0)
			break;
		if (rc == -1) {
			fprintf(stderr, "error from sendfile: %s\n", strerror(errno));
			return -1;
		}
	}
	if (offset != stat_source.st_size) {
		fprintf(stderr, "incomplete transfer from sendfile: %lld of %lld bytes\n", (long long)rc, (long long)stat_source.st_size);
		return -1;
	}
	close(source);
	close(dest);

	// remove source file after copy
	remove(src);

	return 0;
}

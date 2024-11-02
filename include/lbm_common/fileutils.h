#pragma once

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

// create parent directories of a file path
static int create_parent_directories(const char* fname)
{
	char buffer[PATH_MAX];
	strcpy(buffer, fname);
	char* dir = dirname(buffer);
	return mkdir_p(dir, 0777);
}

// create parent directories and then the file
static int create_file(const char* fname)
{
	// return early if the file already exists
	if (fileExists(fname))
		return 0;

	// make sure that the parent directory exists
	create_parent_directories(fname);

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

CYTHON_DIRS=integrators lie_algebras vortices

all:
	for cython_dir in $(CYTHON_DIRS); do \
		echo $$cython_dir; \
		(cd $$cython_dir; python setup.py build_ext --inplace); \
	done

clean:
	for cython_dir in $(CYTHON_DIRS); do \
		(cd $$cython_dir; rm -rf *.so *.c); \
	done

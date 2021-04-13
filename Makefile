.PHONY: all benchmark install install-requirements uninstall test build clean-dist clean-temp test version

all: clean version
	$(MAKE) build
	rm -rf py_ssvad_metrics.egg-info temp build
	$(MAKE) clean-temp

install:
	pip3 install --no-cache-dir --upgrade --find-links dist py-ssvad-metrics

uninstall:
	pip3 uninstall -y py-ssvad-metrics

clean: clean-temp

install-requirements:
	pip3 install -r requirements.txt

test:
	pytests -s

build:
	python3 setup.py bdist_wheel

clean-dist:
	rm -rf dist

clean-temp:
	rm -rf py_ssvad_metrics.egg-info temp build
	find . -name "*.c" -type f -delete

VERSION = $(shell git describe --tag | python3 versioning_git_to_pep440.py)
version:
	echo $(VERSION) | tr -d '\n' > ssvad_metrics/VERSION
	echo "Package version: $(VERSION)"
PACKAGE_NAME=omf

.PHONY: install publish docs coverage lint lint-html graphs test-docs tests

install:
	python setup.py install

publish:
	python setup.py sdist upload

docs:
	cd docs && make html

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=$(PACKAGE_NAME) --cover-html
	open cover/index.html

lint:
	pylint $(PACKAGE_NAME)

lint-html:
	pylint --output-format=html $(PACKAGE_NAME) > pylint.html

graphs:
	pyreverse -my -A -o pdf -p $(PACKAGE_NAME) $(PACKAGE_NAME)/**.py $(PACKAGE_NAME)/**/**.py

test-docs:
	nosetests --logging-level=INFO docs

tests:
	nosetests --logging-level=INFO --with-coverage --cover-package=$(PACKAGE_NAME)
	make lint

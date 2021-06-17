ORG=gmggroup
APP=omf

.PHONY: install publish docs coverage lint lint-html graphs test-docs tests

install:
	python setup.py install

docs:
	cd docs && make html

lint:
	pylint $(APP)

test-docs:
	nosetests --logging-level=INFO docs

tests:
	pytest tests/

docker-build:
	docker build -t $(ORG)/$(APP):latest -f Dockerfile .

docker-build-27:
	docker build -t $(ORG)/$(APP):latest27 -f Dockerfile.27 .

docker-tests: docker-build
	mkdir -p cover
	docker run --rm \
		--name=$(APP)-tests \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/tests:/usr/src/app/tests \
		-v $(shell pwd)/cover:/usr/src/app/cover \
		$(ORG)/$(APP):latest \
		bash -c "pytest --cov=$(APP) --cov-report term --cov-report html:cover/ tests/ && cp .coverage cover/"
	mv -f cover/.coverage ./

docker-tests-27: docker-build-27
	docker run --rm \
		--name=$(APP)-tests \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/tests:/usr/src/app/tests \
		$(ORG)/$(APP):latest27 \
		bash -c "pytest tests/"

docker-lint: docker-build
	docker run --rm \
		--name=$(APP)-tests \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/tests:/usr/src/app/tests \
		-v $(shell pwd)/.pylintrc:/usr/src/app/.pylintrc \
		$(ORG)/$(APP):latest \
		pylint --rcfile=.pylintrc $(APP) tests

docker-docs: docker-build
	docker run --rm \
		--name=$(APP)-tests \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/docs:/usr/src/app/docs \
		$(ORG)/$(APP):latest \
		nosetests --logging-level=INFO docs

publish: docker-build
	mkdir -p dist
	docker run --rm \
		--name=$(APP)-publish \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/dist:/usr/src/app/dist \
		$(ORG)/$(APP) \
		python setup.py sdist bdist_wheel
	pip install twine
	twine upload dist/*

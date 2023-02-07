ORG=gmggroup
APP=omf

.PHONY: install docs lint test-docs tests docker-build docker-tests docker-lint docker-docs publish

install:
	pip install .

docs:
	cd docs && make html

lint:
	pylint $(APP)

test-docs:
	nosetests --logging-level=INFO docs

tests:
	pytest tests/

format:
	black .

docker-build:
	docker build -t $(ORG)/$(APP):latest -f Dockerfile .

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

docker-lint: docker-build
	docker run --rm \
		--name=$(APP)-tests \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/tests:/usr/src/app/tests \
		-v $(shell pwd)/.pylintrc:/usr/src/app/.pylintrc \
		$(ORG)/$(APP):latest \
		pylint --rcfile=.pylintrc $(APP) tests

docker-format: docker-build
	docker run --rm \
		--name=$(APP)-tests \
		-v $(shell pwd)/$(APP):/usr/src/app/$(APP) \
		-v $(shell pwd)/tests:/usr/src/app/tests \
		$(ORG)/$(APP):latest \
		black --check /usr/src/app

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

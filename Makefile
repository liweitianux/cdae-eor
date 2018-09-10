# Path to the virtualenv
VENV ?= venv
REQ ?= requirements.txt


help:
	@echo "venv [VENV=<venv>] [REQ=<requirements.txt>]"
	@echo "    create virtualenv '${VENV}' and install the requirements"
	@echo "    from file '${REQ}'"

# Create virtualenv and install/update the dependencies
# Credit: http://blog.bottlepy.org/2012/07/16/virtualenv-and-makefiles.html
venv: ${VENV}/bin/activate
${VENV}/bin/activate: requirements.txt
	test -d ${VENV} || python3 -m venv ${VENV}
	${VENV}/bin/pip3 install -r ${REQ}
	touch ${VENV}/bin/activate

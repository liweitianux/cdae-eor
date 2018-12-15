#!/bin/sh

PY_VER=`python -V 2>/dev/stdout|awk '{print $2}'|grep ^3 >/dev/null && echo 3`

echo "-lboost_python${PY_VER} -lboost_numpy${PY_VER}"

#!/bin/bash

TEST_EXEC=$1
APP_ARGS=$2
TEST_MODE=$3

echo Exec: ${TEST_EXEC} Args: ${APP_ARGS} Mode: ${TEST_MODE}

RETURN_CODE=1

${TEST_EXEC} ${APP_ARGS}
APP_RET=$?
echo ${TEST_MODE} ${APP_RET}

if [ "${TEST_MODE}" == "FAIL" ]; then
   if [ ${APP_RET} != 0 ]; then
      RETURN_CODE=0
   fi
else
   RETURN_CODE=${APP_RET}
fi

exit ${RETURN_CODE}


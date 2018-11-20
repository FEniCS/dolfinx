#!/bin/bash
curl -u ${CIRCLE_API_USER_TOKEN} \
     -d build_parameters[CIRCLE_JOB]=build-and-push-docker-images \
     https://circleci.com/api/v1.1/project/github/FEniCS/dolfinx/tree/master

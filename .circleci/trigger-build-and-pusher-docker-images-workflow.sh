#!/bin/bash
curl -u ${CIRCLECI_TOKEN}: -X POST --header "Content-Type: application/json" -d '{
 "parameters": {
   "build-and-push-docker-images": true
 }
}' https://circleci.com/api/v2/project/gh/FEniCS/dolfinx/pipeline

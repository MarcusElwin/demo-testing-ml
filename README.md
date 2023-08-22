# demo-testing-ml
Repository to show case simple and minimal testing of ML systems part of demo for KaggleX

## Type of tests

### Pre-training tests
__TODO__

### Post-training tests
__TODO__

### Data drift tests
__TODO__

## Local Setup
1) Install [vs-code]()
2) Install the [dev-containers](https://code.visualstudio.com/docs/devcontainers/tutorial) extension
3) Use the `DockerFile` in this repository and `Re-open` as the container: 
![Open dev container](image.png)

## Running tests
To run all tests in this repository run the below:
```sh
python -m pytest --disable-pytest-warnings src/ --no-header -v
```
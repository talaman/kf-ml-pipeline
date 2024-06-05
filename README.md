# kf-ml-pipeline


## Install Kubeflow

Windows example. For other OS, please refer to the official [Kubeflow documentation](https://www.kubeflow.org/docs/started/getting-started/).

### Requirements

- [Docker](https://www.docker.com/) with Kubernetes enabled
- Kubectl

### Install Kustomize

If you do not already have it, you can get kustomize with docker and the following command:

```bash
docker pull registry.k8s.io/kustomize/kustomize:v5.0.0
docker run registry.k8s.io/kustomize/kustomize:v5.0.0 version
```

### Install Kubeflow Pipelines

To install Kubeflow Pipelines using manifests from Kubeflow Pipelines’s GitHub repository, run these commands:

```bash
set PIPELINE_VERSION=2.2.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=%PIPELINE_VERSION%"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=%PIPELINE_VERSION%"
```

### Access Kubeflow Pipelines

To access the Kubeflow Pipelines UI, run the following command:

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 9000:80
```


Then, open a web browser and go to `http://localhost:9000/`.

### Uninstall Kubeflow Pipelines

To uninstall Kubeflow Pipelines, run the following commands:

```bash
set PIPELINE_VERSION=2.2.0
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=%PIPELINE_VERSION%"
kubectl delete -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=%PIPELINE_VERSION%"

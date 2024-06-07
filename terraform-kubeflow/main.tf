provider "kubernetes" {
  config_path = "~/.kube/config"
}

resource "null_resource" "kubeflow_cluster_scoped" {
  provisioner "local-exec" {
    command = "kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl delete -k github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.2.0"
  }
}

resource "null_resource" "wait_for_crd" {
  provisioner "local-exec" {
    command = "kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io"
  }

  triggers = {
    always_run = "${timestamp()}"
  }

  depends_on = [null_resource.kubeflow_cluster_scoped]
}

resource "null_resource" "kubeflow_platform_agnostic" {
  provisioner "local-exec" {
    command = "kubectl apply -k github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.2.0"
  }

  provisioner "local-exec" {
    when    = destroy
    command = "kubectl delete -k github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.2.0"
  }

  depends_on = [null_resource.wait_for_crd]
}

resource "kubernetes_service" "ml_pipeline_ui_lb" {
  metadata {
    name      = "ml-pipeline-ui-lb"
    namespace = "kubeflow"
  }

  spec {
    type = "LoadBalancer"

    port {
      port        = 9000
      target_port = 3000
    }

    selector = {
      app = "ml-pipeline-ui"
    }
  }

  depends_on = [null_resource.kubeflow_platform_agnostic]
}


## EKS cluster
### Creating
```
eksctl create cluster -f cluster4gaudi.yaml
```

### Gaudi device plugin
```
kubectl apply -f habana-k8s-device-plugin.yaml
```

### Test Gaudi device access
```
kubectl apply -f test-gaudi.yaml
```

### Taint Gaudi nodes
```

```

## Kuberay

```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator
kubectl get pods
```

```
kubectl get raycluster
```


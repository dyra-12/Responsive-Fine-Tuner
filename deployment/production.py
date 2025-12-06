import os
import sys
import logging
import subprocess
import yaml
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import docker
import paramiko
import boto3
import google.cloud.resourcemanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionDeployer:
    """Production deployment manager for RFT"""
    
    def __init__(self, config_path: str = "deployment/production.yaml"):
        self.config = self._load_config(config_path)
        self.deployment_history = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def deploy_to_docker(self, build_args: Dict[str, str] = None) -> Dict[str, Any]:
        """Deploy using Docker"""
        try:
            logger.info("Starting Docker deployment...")
            
            # Build Docker image
            build_cmd = ["docker", "build", "-t", "rft-app:latest", "."]
            if build_args:
                for key, value in build_args.items():
                    build_cmd.extend(["--build-arg", f"{key}={value}"])
            
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Docker build failed: {result.stderr}")
            
            # Run container
            run_cmd = [
                "docker", "run", "-d",
                "-p", f"{self.config['port']}:7860",
                "--name", "rft-app",
                "--restart", "unless-stopped",
                "-v", "rft-data:/app/data",
                "-v", "rft-models:/app/models",
                "rft-app:latest"
            ]
            
            result = subprocess.run(run_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Docker run failed: {result.stderr}")
            
            deployment_info = {
                "type": "docker",
                "status": "success",
                "container_id": result.stdout.strip(),
                "port": self.config['port'],
                "timestamp": datetime.now().isoformat()
            }
            
            self.deployment_history.append(deployment_info)
            logger.info("Docker deployment successful")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return {"type": "docker", "status": "error", "error": str(e)}
    
    def deploy_to_kubernetes(self, namespace: str = "default") -> Dict[str, Any]:
        """Deploy to Kubernetes cluster"""
        try:
            logger.info("Starting Kubernetes deployment...")
            
            # Create deployment YAML
            deployment_yaml = self._generate_k8s_deployment_yaml(namespace)
            
            # Apply configuration
            with open("deployment/k8s-deployment.yaml", "w") as f:
                yaml.dump(deployment_yaml, f)
            
            # Apply to cluster
            apply_cmd = ["kubectl", "apply", "-f", "deployment/k8s-deployment.yaml"]
            result = subprocess.run(apply_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Kubernetes deployment failed: {result.stderr}")
            
            # Create service
            service_yaml = self._generate_k8s_service_yaml(namespace)
            with open("deployment/k8s-service.yaml", "w") as f:
                yaml.dump(service_yaml, f)
            
            apply_cmd = ["kubectl", "apply", "-f", "deployment/k8s-service.yaml"]
            result = subprocess.run(apply_cmd, capture_output=True, text=True)
            
            deployment_info = {
                "type": "kubernetes",
                "status": "success",
                "namespace": namespace,
                "service_name": "rft-service",
                "timestamp": datetime.now().isoformat()
            }
            
            self.deployment_history.append(deployment_info)
            logger.info("Kubernetes deployment successful")
            return deployment_info
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return {"type": "kubernetes", "status": "error", "error": str(e)}
    
    def _generate_k8s_deployment_yaml(self, namespace: str) -> Dict[str, Any]:
        """Generate Kubernetes deployment YAML"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "rft-deployment",
                "namespace": namespace,
                "labels": {"app": "rft"}
            },
            "spec": {
                "replicas": self.config.get('replicas', 2),
                "selector": {"matchLabels": {"app": "rft"}},
                "template": {
                    "metadata": {"labels": {"app": "rft"}},
                    "spec": {
                        "containers": [{
                            "name": "rft-app",
                            "image": "rft-app:latest",
                            "ports": [{"containerPort": 7860}],
                            "env": [
                                {"name": "MODEL_CACHE_DIR", "value": "/app/models"},
                                {"name": "DATA_DIR", "value": "/app/data"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "2Gi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "1000m"
                                }
                            },
                            "volumeMounts": [
                                {"name": "data-volume", "mountPath": "/app/data"},
                                {"name": "models-volume", "mountPath": "/app/models"}
                            ]
                        }],
                        "volumes": [
                            {"name": "data-volume", "emptyDir": {}},
                            {"name": "models-volume", "emptyDir": {}}
                        ]
                    }
                }
            }
        }
    
    def deploy_to_cloud(self, provider: str = "aws") -> Dict[str, Any]:
        """Deploy to cloud provider"""
        try:
            logger.info(f"Starting {provider.upper()} deployment...")
            
            if provider == "aws":
                return self._deploy_to_aws()
            elif provider == "azure":
                return self._deploy_to_azure()
            elif provider == "gcp":
                return self._deploy_to_gcp()
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
                
        except Exception as e:
            logger.error(f"Cloud deployment failed: {e}")
            return {"type": provider, "status": "error", "error": str(e)}
    
    def _deploy_to_aws(self) -> Dict[str, Any]:
        """Deploy to AWS ECS/EKS"""
        # Implementation for AWS deployment
        return {"type": "aws", "status": "success", "message": "AWS deployment placeholder"}
    
    def rollback_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        try:
            # Find deployment in history
            deployment = next((d for d in self.deployment_history if d.get('id') == deployment_id), None)
            if not deployment:
                return {"status": "error", "message": "Deployment not found"}
            
            logger.info(f"Rolling back deployment: {deployment_id}")
            
            # Implementation depends on deployment type
            if deployment['type'] == 'docker':
                return self._rollback_docker(deployment)
            elif deployment['type'] == 'kubernetes':
                return self._rollback_kubernetes(deployment)
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            "active_deployments": len(self.deployment_history),
            "last_deployment": self.deployment_history[-1] if self.deployment_history else None,
            "system_status": self._check_system_health()
        }
# 🚀 Deployment Guide

This guide covers various deployment options for the Credit Card Fraud Detection System.

## 📋 Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- Git
- Required API keys (for cloud platforms)

## 🐳 Docker Deployment

### Local Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t fraud-detection .
   ```

2. **Run with Docker Compose (Recommended):**
   ```bash
   docker-compose up -d
   ```

3. **Access the services:**
   - API: http://localhost:8000
   - Dashboard: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

### Docker Commands

```bash
# Run API only
docker run -p 8000:8000 fraud-detection

# Run dashboard only
docker run -p 8501:8501 fraud-detection streamlit run src/dashboard.py

# Run with custom environment
docker run -p 8000:8000 -e ENVIRONMENT=production fraud-detection
```

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Install Heroku CLI:**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Heroku app:**
   ```bash
   heroku create your-fraud-detection-app
   ```

3. **Set environment variables:**
   ```bash
   heroku config:set ENVIRONMENT=production
   heroku config:set LOG_LEVEL=INFO
   ```

4. **Deploy:**
   ```bash
   git push heroku main
   ```

5. **Open the app:**
   ```bash
   heroku open
   ```

### AWS Deployment

#### Option 1: AWS ECS with Fargate

1. **Create ECR repository:**
   ```bash
   aws ecr create-repository --repository-name fraud-detection
   ```

2. **Build and push image:**
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t fraud-detection .
   docker tag fraud-detection:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/fraud-detection:latest
   ```

3. **Deploy to ECS:**
   ```bash
   # Use AWS Console or AWS CLI to create ECS cluster and service
   ```

#### Option 2: AWS Lambda with API Gateway

1. **Package for Lambda:**
   ```bash
   pip install -r requirements.txt -t package/
   cd package
   zip -r ../lambda-deployment.zip .
   ```

2. **Deploy to Lambda:**
   ```bash
   aws lambda create-function \
     --function-name fraud-detection \
     --runtime python3.9 \
     --handler src.api.handler \
     --zip-file fileb://lambda-deployment.zip
   ```

### Google Cloud Platform

#### Option 1: Cloud Run

1. **Enable Cloud Run API:**
   ```bash
   gcloud services enable run.googleapis.com
   ```

2. **Build and deploy:**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/fraud-detection
   gcloud run deploy fraud-detection \
     --image gcr.io/PROJECT-ID/fraud-detection \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

#### Option 2: App Engine

1. **Create app.yaml:**
   ```yaml
   runtime: python39
   entrypoint: gunicorn -b :$PORT src.api:app
   
   env_variables:
     ENVIRONMENT: production
   ```

2. **Deploy:**
   ```bash
   gcloud app deploy
   ```

### Azure Deployment

#### Option 1: Azure Container Instances

1. **Build and push to Azure Container Registry:**
   ```bash
   az acr build --registry <registry-name> --image fraud-detection .
   ```

2. **Deploy:**
   ```bash
   az container create \
     --resource-group <resource-group> \
     --name fraud-detection \
     --image <registry-name>.azurecr.io/fraud-detection:latest \
     --ports 8000 \
     --dns-name-label fraud-detection
   ```

#### Option 2: Azure App Service

1. **Create App Service:**
   ```bash
   az webapp create --resource-group <resource-group> --plan <plan-name> --name fraud-detection
   ```

2. **Deploy:**
   ```bash
   az webapp deployment source config-zip --resource-group <resource-group> --name fraud-detection --src <zip-file>
   ```

## 🔧 Environment Configuration

### Environment Variables

```bash
# Required
ENVIRONMENT=production
LOG_LEVEL=INFO

# Optional
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://localhost:6379
API_KEY=your-secret-key
```

### Configuration Files

Create `.env` file for local development:

```env
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///fraud_detection.db
```

## 📊 Monitoring and Logging

### Health Checks

- API Health: `GET /health`
- Dashboard Health: Check Streamlit status
- Model Health: `GET /model/health`

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Metrics

- Request count
- Response time
- Error rate
- Model prediction accuracy

## 🔒 Security Considerations

### API Security

1. **Authentication:**
   ```python
   from fastapi import HTTPException, Depends
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   async def verify_token(token: str = Depends(security)):
       if not is_valid_token(token):
           raise HTTPException(status_code=401)
   ```

2. **Rate Limiting:**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   ```

3. **CORS Configuration:**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

## 🚀 Performance Optimization

### Production Settings

1. **Gunicorn Configuration:**
   ```bash
   gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Streamlit Configuration:**
   ```bash
   streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.maxUploadSize 200
   ```

3. **Database Optimization:**
   - Use connection pooling
   - Implement caching with Redis
   - Optimize database queries

## 📈 Scaling

### Horizontal Scaling

1. **Load Balancer Configuration:**
   ```nginx
   upstream fraud_detection {
       server 127.0.0.1:8000;
       server 127.0.0.1:8001;
       server 127.0.0.1:8002;
   }
   ```

2. **Auto-scaling:**
   - Set up auto-scaling groups
   - Configure health checks
   - Monitor resource usage

### Vertical Scaling

1. **Resource Allocation:**
   - Increase CPU and memory
   - Optimize model inference
   - Use GPU acceleration for ML models

## 🔄 CI/CD Pipeline

### GitHub Actions

The project includes a complete CI/CD pipeline:

1. **Testing:** Automated testing on every push
2. **Building:** Docker image building
3. **Deployment:** Automatic deployment to cloud platforms

### Manual Deployment

```bash
# Run tests
pytest tests/

# Build Docker image
docker build -t fraud-detection .

# Deploy
docker push <registry>/fraud-detection:latest
```

## 🆘 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Find process using port
   lsof -i :8000
   # Kill process
   kill -9 <PID>
   ```

2. **Docker build fails:**
   ```bash
   # Clean Docker cache
   docker system prune -a
   # Rebuild
   docker build --no-cache -t fraud-detection .
   ```

3. **Model loading errors:**
   ```bash
   # Ensure models are trained
   python src/main.py
   # Check model files exist
   ls -la models/
   ```

### Debug Mode

```bash
# Run with debug logging
ENVIRONMENT=development LOG_LEVEL=DEBUG python src/api.py

# Run dashboard in debug mode
streamlit run src/dashboard.py --logger.level debug
```

## 📞 Support

For deployment issues:

1. Check the logs: `docker logs <container-id>`
2. Verify environment variables
3. Test locally before deploying
4. Check network connectivity
5. Review security group settings

---

**Happy Deploying! 🚀**


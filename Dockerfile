# -----------------------------
# Dockerfile for Credit Risk Project
# -----------------------------

# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set work directory inside container
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Create necessary folders in container
RUN mkdir -p data models figures results

# Expose Jupyter Notebook port (optional for research/demo)
EXPOSE 8888

# Default command: run credit risk pipeline directly
# You can also run Jupyter Notebook if desired
CMD ["python", "scripts/credit_risk_pipeline.py"]
# If you want Jupyter Notebook instead, comment the line above and uncomment below
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
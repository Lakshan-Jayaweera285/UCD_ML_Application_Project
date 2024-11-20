FROM python:3.9

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose port
EXPOSE 8000

#FastAPI run command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

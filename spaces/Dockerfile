FROM python:3.11-slim

# HuggingFace Spaces requires user ID 1000
RUN useradd -m -u 1000 user

WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user server.py .

USER user
ENV HOME=/home/user

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]

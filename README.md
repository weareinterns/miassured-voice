# Healio Voice - Gemini AI WebSocket Server

A WebSocket server that connects browser audio to Gemini AI and returns audio responses.

## Overview

This application:
1. Accepts audio data via WebSocket from the browser (client handles microphone)
2. Sends the audio data to Gemini AI
3. Returns Gemini's audio responses back to the browser via WebSocket

## Architecture

The application runs two services:
- A FastAPI server on port 8080 for health checks and a WebSocket path
- A standalone WebSocket server on port 8765 for direct WebSocket connections

## Deployment to Koyeb

### Deployment Configuration

1. Push this code to a GitHub repository
2. In Koyeb dashboard, create a new app
3. Choose GitHub as the deployment method
4. Select your repository and branch
5. Configure environment variables (see below)
6. Set the health check settings:
   - Path: `/health`
   - Initial delay: 5 seconds
   - Interval: 5 seconds 
   - Timeout: 3 seconds
   - Threshold: 1

### Environment Variables

- `PORT`: The port for the HTTP server (default: 8080)
- `GEMINI_API_KEY`: Your Google Gemini API key

## Important Notes for Client Connection

When connecting from your browser client, you have two options:

### Option 1: Through the FastAPI WebSocket endpoint (Recommended for Koyeb)

```javascript
// Connect through the FastAPI WebSocket endpoint
const ws = new WebSocket('wss://your-koyeb-app.koyeb.app/ws');
```

### Option 2: Direct WebSocket connection

```javascript
// Direct WebSocket connection (may not work with all proxies)
const ws = new WebSocket('wss://your-koyeb-app.koyeb.app:8765');
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable for Gemini API Key (optional, fallback in code)
export GEMINI_API_KEY="your-api-key"

# Run the server
python HealioVoice.py
```

The HTTP server will start on http://localhost:8080
The WebSocket server will be available at:
- ws://localhost:8765 (direct WebSocket)
- ws://localhost:8080/ws (FastAPI WebSocket endpoint) 
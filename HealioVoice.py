import asyncio
import websockets
import json
import base64
import traceback
import io
import wave
import struct
import requests
import urllib.request
import uuid
import sys
import os
import time
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import threading

from google import genai
from google.genai import types

# Audio settings
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000

# API keys list - hardcoded for private repo
API_KEYS = ["AIzaSyDxt6OFilvmOGCohVU-Lk110L4vcKpt8Yw",
"AIzaSyDk4BK6cb7dakuPOTSApVgl6qaCZ-DMZGQ",
"AIzaSyACs4pwIhUbQT1iBVF7p7pMYyoIjSR77fk",
"AIzaSyBoZwMjtQbC-UEPK-a5XRAAtNik1DYY_8c",
"AIzaSyA5nJsP5e_MCEM3i0UE12-PKTRlk8-MaWA",
"AIzaSyCE7Nzki1uiDqysW1I6fzTXWvXj9QhDKP4",
"AIzaSyBwCXA22KuqIC59CvLJVAYiuoNutoVhdno",
"AIzaSyChnACA8hhiaqUuLK_cn1bHlHOeNfNDpL4",
"AIzaSyBzSWAYSeaYRp7QM66Zvy5isVwmAPgPBkk",
"AIzaSyCbJcF5npBxMk0WjDsG1Yd5EyYmnEjCYic",
"AIzaSyC9OLX_jIz256TUsA_BHkFuMTs7b2V6GBM",
"AIzaSyC4RqGpaaacbQNdfZE8pjcqv7hjLWIpHEA",
"AIzaSyDgcEmCmO6dcEr9dFl5ylVE7bPhHKqw3xI",
"AIzaSyBFROU2GlFp_oy22_Ff5v8HIdprh6xDIMc",
"AIzaSyCY0IXljzIUbto9cALbZtcZpN6VZxR0l_I",
"AIzaSyBoW809DQokDLEcGQOsvi2-2vMqlqwT4jA",
"AIzaSyAPN991PlxeCkqyffvHR6cEvca5UjeJvYI",
"AIzaSyAtv3dUZyWopNCrb8XDqnw9_GGvFWdfHK8",
"AIzaSyBZGVTlKeqJeul0lK2ossg1aqUxUUyRUyA",
"AIzaSyAEjmoTDTdGk-4OVAwT3gcP5AztvpSNciw"]

# API key rotation settings
RANDOM_KEY_PER_SESSION = True  # Set to True to use a random key for each new session
ROTATION_INTERVAL = 300  # Rotation interval in seconds (5 minutes)

# Current API key index and last rotation time
current_api_key_index = 0
last_rotation_time = time.time()
api_key_lock = threading.Lock()

# Get current API key
def get_current_api_key():
    global current_api_key_index, last_rotation_time
    current_time = time.time()
    
    # For new sessions with random assignment enabled, bypass the rotation logic
    if RANDOM_KEY_PER_SESSION:
        return get_random_api_key()
    
    with api_key_lock:
        # Check if it's time to rotate API key (every 5 minutes)
        if current_time - last_rotation_time >= ROTATION_INTERVAL:
            # Instead of sequential rotation, pick a random key
            current_api_key_index = random.randint(0, len(API_KEYS) - 1)
            last_rotation_time = current_time
            print(f"API key rotated to index {current_api_key_index}")
        
        return API_KEYS[current_api_key_index]

# Get a random API key from the list
def get_random_api_key():
    return API_KEYS[random.randint(0, len(API_KEYS) - 1)]

# Model and config
MODEL = "models/gemini-2.0-flash-live-001"

# Initial prompt URL
PROMPT_URL = "https://gist.githubusercontent.com/shudveta/6826b7063ca934a799f191a14797e05f/raw/83192b8d1359b96a6989c3c001306b028baf861a/prompt-voice.txt"

# Active client sessions
active_sessions = {}

# Check Python version - needed for websockets compatibility
PY_VERSION = sys.version_info
IS_PY313_PLUS = PY_VERSION >= (3, 13)

# Create FastAPI app for health checks
app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/ping")
async def ping():
    return {"status": "active", "timestamp": str(uuid.uuid4()), "sessions": len(active_sessions)}

@app.get("/")
async def root():
    return {"message": "Healio Voice API is running. Connect via WebSocket for audio interaction."}

# Add a FastAPI WebSocket route for Koyeb compatibility
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt received at /ws endpoint")
    await websocket.accept()
    print("WebSocket connection accepted at /ws endpoint")
    handler = AudioSessionHandler(websocket)
    await handler.start_session()

# Fetch the prompt content
def load_prompt():
    try:
        with urllib.request.urlopen(PROMPT_URL) as response:
            prompt_text = response.read().decode()
            print(f"Successfully loaded prompt: {len(prompt_text)} characters")
            return prompt_text
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return ""

# Load the prompt at startup
INITIAL_PROMPT = load_prompt()

# Basic Gemini configuration
CONFIG = types.LiveConnectConfig(
    response_modalities=["audio"],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Leda")
        )
    ),
    tools=[{
        "google_search_retrieval": {}
    }],
)

# Background task to rotate API keys
async def api_key_rotator():
    while True:
        await asyncio.sleep(ROTATION_INTERVAL)  # Use the rotation interval constant
        with api_key_lock:
            global current_api_key_index, last_rotation_time
            # Use random selection instead of sequential
            current_api_key_index = random.randint(0, len(API_KEYS) - 1)
            last_rotation_time = time.time()
            current_key = API_KEYS[current_api_key_index]
            print(f"API key automatically rotated to random index {current_api_key_index}")

class AudioSessionHandler:
    def __init__(self, websocket):
        self.session_id = str(uuid.uuid4())[:8]  # Generate short unique session ID
        self.websocket = websocket
        # Get current API key at session initialization
        self.api_key = get_current_api_key()
        if RANDOM_KEY_PER_SESSION:
            print(f"[{self.session_id}] Using random API key for new session")
        else:
            print(f"[{self.session_id}] Using API key index {current_api_key_index}")
        self.client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=self.api_key)
        self.session = None
        self.running = True
        self.out_queue = asyncio.Queue(maxsize=5)
        self.audio_chunks = []
        self.buffer_size = 0
        self.max_buffer_size = 192000
        self.initial_prompt = INITIAL_PROMPT
        self.connection_closed = False  # Flag to track WebSocket connection state
        
        # Register this session
        active_sessions[self.session_id] = self
        print(f"New session created: {self.session_id}, Total active sessions: {len(active_sessions)}")
        
    async def start_session(self):
        try:
            # Connect to the Gemini API using proper async with syntax
            print(f"[{self.session_id}] Connecting to Gemini API")
            async with self.client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print(f"[{self.session_id}] Successfully connected to Gemini API")
                
                # Send initial prompt if available
                if self.initial_prompt:
                    print(f"[{self.session_id}] Sending initial prompt to Gemini")
                    # Fix for the API method issue - use try/except for each possible method
                    try:
                        # Try to send as plain input with end_of_turn parameter
                        await self.session.send(input=self.initial_prompt, end_of_turn=True)
                        print(f"[{self.session_id}] Used session.send with end_of_turn")
                    except TypeError as e:
                        print(f"[{self.session_id}] Error with session.send + end_of_turn: {e}")
                        # Try without end_of_turn parameter
                        try:
                            await self.session.send(input=self.initial_prompt)
                            print(f"[{self.session_id}] Used session.send without end_of_turn")
                        except Exception as e2:
                            print(f"[{self.session_id}] Error with simple session.send: {e2}")
                            # Last resort - try different methods
                            try:
                                await self.session.send_client_content(self.initial_prompt)
                                print(f"[{self.session_id}] Used session.send_client_content")
                            except Exception as e3:
                                print(f"[{self.session_id}] Failed to send initial prompt: {e3}")
                    self.initial_prompt = ""  # Clear prompt after sending
                
                # Send status message to client - FIX HERE: use send_text for FastAPI WebSocket
                await self.safe_send_text(json.dumps({
                    "type": "status",
                    "data": f"Connected to Dr. Healio (Session: {self.session_id})"
                }))
                
                # Start all the tasks
                send_task = asyncio.create_task(self.send_audio())
                receive_task = asyncio.create_task(self.receive_audio())
                process_task = asyncio.create_task(self.process_browser_messages())
                
                # Wait for tasks to complete
                await asyncio.gather(send_task, receive_task, process_task)
            
        except websockets.exceptions.ConnectionClosed as e:
            # Normal connection closure, not an error
            print(f"[{self.session_id}] WebSocket connection closed: {e}")
            self.connection_closed = True
        except WebSocketDisconnect as e:
            # FastAPI WebSocket disconnect
            print(f"[{self.session_id}] FastAPI WebSocket connection closed: {e}")
            self.connection_closed = True
        except Exception as e:
            print(f"[{self.session_id}] Error in session: {e}")
            traceback.print_exc()
            # Don't check websocket.closed - use try/except instead
            try:
                # FIX HERE: use send_text for FastAPI WebSocket
                await self.safe_send_text(json.dumps({
                    "type": "status",
                    "data": f"Error: {str(e)}"
                }))
            except:
                # Connection likely already closed
                pass
        finally:
            # Clean up
            self.connection_closed = True
            self.running = False
            print(f"[{self.session_id}] Session ended")
            if self.session_id in active_sessions:
                del active_sessions[self.session_id]
                print(f"Removed session {self.session_id}, Remaining sessions: {len(active_sessions)}")
                
    async def process_browser_messages(self):
        try:
            while self.running and self.session and not self.connection_closed:
                # Get message from browser - FIX HERE: use receive_text for FastAPI WebSocket
                message = await self.websocket.receive_text()
                data = json.loads(message)
                
                if data["type"] == "audio":
                    # Process audio from browser
                    audio_bytes = base64.b64decode(data["data"])
                    await self.out_queue.put({"data": audio_bytes, "mime_type": "audio/pcm"})
                elif data["type"] == "ping":
                    # Respond to ping messages from client
                    await self.safe_send_text(json.dumps({
                        "type": "pong",
                        "data": "pong"
                    }))
                elif data["type"] == "stop":
                    self.running = False
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            self.running = False
            self.connection_closed = True
        except WebSocketDisconnect:
            self.running = False
            self.connection_closed = True
        except Exception as e:
            print(f"[{self.session_id}] Error processing browser message: {e}")
            traceback.print_exc()
            self.running = False
            
    async def send_audio(self):
        try:
            while self.running and self.session:
                msg = await self.out_queue.get()
                # Try the most basic send method which should work across versions
                try:
                    await self.session.send(input=msg)
                except TypeError as e:
                    print(f"[{self.session_id}] TypeError with session.send in send_audio: {e}")
                    # Try alternative methods
                    try:
                        # Try other methods if available
                        if hasattr(self.session, 'send_realtime_input') and callable(getattr(self.session, 'send_realtime_input')):
                            await self.session.send_realtime_input(msg)
                            print(f"[{self.session_id}] Used send_realtime_input")
                        elif hasattr(self.session, 'send_media_input') and callable(getattr(self.session, 'send_media_input')):
                            await self.session.send_media_input(msg)
                            print(f"[{self.session_id}] Used send_media_input")
                        else:
                            # Last resort: try without input parameter
                            await self.session.send(msg)
                            print(f"[{self.session_id}] Used session.send without input parameter")
                    except Exception as e2:
                        print(f"[{self.session_id}] All send methods failed: {e2}")
                        raise
        except websockets.exceptions.ConnectionClosed:
            self.running = False
            print(f"[{self.session_id}] WebSocket connection closed during audio send")
        except WebSocketDisconnect:
            self.running = False
            print(f"[{self.session_id}] FastAPI WebSocket connection closed during audio send")
        except Exception as e:
            print(f"[{self.session_id}] Error sending audio: {e}")
            traceback.print_exc()
            self.running = False
                
    async def receive_audio(self):
        try:
            last_send_time = asyncio.get_event_loop().time()
            sequence_number = 0
            total_silence_padding = 0.01  # Very short silence between segments (10ms)
            chunk_batch = []
            
            while self.running and self.session and not self.connection_closed:
                turn = self.session.receive()
                first_in_turn = True
                
                async for response in turn:
                    if not self.running or self.connection_closed:
                        break
                        
                    if data := response.data:
                        # Buffer the audio data with timing information
                        current_time = asyncio.get_event_loop().time()
                        self.audio_chunks.append(data)
                        self.buffer_size += len(data)
                        
                        # Collect chunks with minimal buffering - just enough to ensure we have complete words
                        # Send small chunks more frequently (32KB or 250ms)
                        buffer_full = self.buffer_size >= 32000
                        time_elapsed = current_time - last_send_time
                        
                        if buffer_full or (time_elapsed > 0.25 and self.buffer_size > 0):
                            sequence_number += 1
                            
                            # Add this chunk to the batch with its sequence info
                            chunk_batch.append({
                                'sequence': sequence_number,
                                'is_first': first_in_turn,
                                'is_turn_end': False,
                                'timing': current_time  # Use for relative timing
                            })
                            
                            await self.send_buffer(
                                continuous=not first_in_turn,
                                sequence=sequence_number,
                                is_first=first_in_turn,
                                is_turn_end=False,
                                silence_padding=total_silence_padding
                            )
                            
                            last_send_time = current_time
                            first_in_turn = False
                        
                    if text := response.text:
                        # Send text to browser - FIX: use send_text for FastAPI WebSocket
                        await self.safe_send_text(json.dumps({
                            "type": "text",
                            "data": text
                        }))
                
                # Send any remaining audio at end of turn
                if self.audio_chunks and not self.connection_closed:
                    sequence_number += 1
                    
                    # Add final chunk to batch
                    chunk_batch.append({
                        'sequence': sequence_number,
                        'is_first': False,
                        'is_turn_end': True,
                        'timing': asyncio.get_event_loop().time()
                    })
                    
                    # End of turn gets slightly longer silence padding
                    await self.send_buffer(
                        continuous=False,
                        sequence=sequence_number,
                        is_first=False,
                        is_turn_end=True,
                        silence_padding=0.2  # 200ms pause at end of turn
                    )
                    
                    # Send batch info to help client sync audio
                    if chunk_batch:
                        # FIX: use send_text for FastAPI WebSocket
                        await self.safe_send_text(json.dumps({
                            "type": "audio_batch",
                            "data": chunk_batch,
                            "turn_end": True
                        }))
                        chunk_batch = []  # Reset for next turn
                    
                    # Small silence between turns
                    await asyncio.sleep(0.05)
                
        except Exception as e:
            print(f"[{self.session_id}] Error receiving audio: {e}")
            traceback.print_exc()
            self.running = False
    
    async def safe_send_text(self, text):
        """Send text to the client with connection state checking"""
        if self.connection_closed:
            return
            
        try:
            await self.websocket.send_text(text)
        except websockets.exceptions.ConnectionClosed:
            self.connection_closed = True
            self.running = False
        except WebSocketDisconnect:
            self.connection_closed = True
            self.running = False
        except RuntimeError as e:
            # This likely means the connection is already closed
            if "after sending 'websocket.close'" in str(e):
                self.connection_closed = True
                self.running = False
            else:
                raise
        except Exception as e:
            print(f"[{self.session_id}] Error in safe_send_text: {e}")
            self.connection_closed = True
            self.running = False

    async def send_buffer(self, continuous=False, sequence=0, is_first=False, is_turn_end=False, silence_padding=0.01):
        """Send the current audio buffer as WAV with silence padding"""
        if not self.audio_chunks or self.connection_closed:
            return
            
        try:
            # Make a copy of chunks and clear buffer before processing
            chunks_to_send = self.audio_chunks.copy()
            self.audio_chunks = []
            self.buffer_size = 0
            
            # Combine all chunks 
            combined_data = b''.join(chunks_to_send)
            
            # Skip very small chunks that might cause problems
            if len(combined_data) < 800:
                print(f"[{self.session_id}] Skipping tiny audio chunk: {len(combined_data)} bytes")
                return
            
            # Convert to WAV
            wav_bytes = self.pcm_to_wav(combined_data, RECEIVE_SAMPLE_RATE)
            
            # Check connection state before sending
            if self.connection_closed:
                return
                
            # Send to client with enhanced metadata
            await self.safe_send_text(json.dumps({
                "type": "audio",
                "data": base64.b64encode(wav_bytes).decode(),
                "continuous": continuous,
                "sequence": sequence,
                "is_first": is_first,
                "is_turn_end": is_turn_end,
                "silence_padding": silence_padding,  # Tell client how much silence to add
                "size": len(wav_bytes),
                "timestamp": asyncio.get_event_loop().time()
            }))
            
            print(f"[{self.session_id}] Sent audio #{sequence}: {len(wav_bytes)} bytes, continuous: {continuous}, padding: {silence_padding}s")
            
        except Exception as e:
            print(f"[{self.session_id}] Error sending buffer: {e}")
            traceback.print_exc()
            self.running = False
    
    def pcm_to_wav(self, pcm_data, sample_rate):
        """Convert raw PCM audio data to WAV format with proper headers"""
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        
        wav_io.seek(0)
        return wav_io.read()

async def handle_connection(websocket):
    # Create new handler for each connection
    handler = AudioSessionHandler(websocket)
    await handler.start_session()

async def websocket_server():
    # Get port from environment variable for Koyeb deployment
    ws_port = 8765  # Use a separate port for WebSockets
    host = "0.0.0.0"  # Listen on all interfaces for cloud deployment
    
    try:
        # For Python 3.13+, use a different approach to handle the server
        if IS_PY313_PLUS:
            print(f"Running on Python 3.13+ - using compatible websockets configuration")
            # Create the websocket server with simpler parameters
            try:
                server = websockets.serve(handle_connection, host, ws_port)
                await server
                print(f"WebSocket server started on ws://{host}:{ws_port}")
                print(f"Server is ready to accept multiple client connections")
                await asyncio.Future()  # Run forever
            except Exception as e:
                print(f"Error with websockets server: {e}")
                traceback.print_exc()
                
                # Try with even simpler parameters
                print("Trying alternate websockets setup")
                server = await websockets.server.serve(handle_connection, host, ws_port)
                print(f"WebSocket server started on ws://{host}:{ws_port} (alternate method)")
                print(f"Server is ready to accept multiple client connections")
                await asyncio.Future()  # Run forever
        else:
            # Standard approach for older Python versions
            server = await websockets.serve(handle_connection, host, ws_port)
            print(f"WebSocket server started on ws://{host}:{ws_port}")
            print(f"Server is ready to accept multiple client connections")
            await asyncio.Future()  # Run forever
    except Exception as e:
        print(f"Error starting server: {e}")
        traceback.print_exc()

def run_fastapi():
    # Get port from environment variable for Koyeb deployment
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

def start_websocket_server():
    asyncio.run(websocket_server())

async def main():
    # Start API key rotation task
    rotation_task = asyncio.create_task(api_key_rotator())
    
    # Start WebSocket server
    await websocket_server()

if __name__ == "__main__":
    # Print info about API key rotation
    print(f"Starting with {len(API_KEYS)} API keys")
    print(f"API key rotation settings:")
    print(f"  - Random key per session: {'Enabled' if RANDOM_KEY_PER_SESSION else 'Disabled'}")
    print(f"  - Rotation interval: {ROTATION_INTERVAL} seconds ({ROTATION_INTERVAL/60} minutes)")
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("HealioVoice:app", host="0.0.0.0", port=port, reload=False)
    
    # Start websocket server in a separate thread (including API key rotation)
    websocket_thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    websocket_thread.start()
    
    # Run FastAPI in the main thread - this will be used for health checks
    run_fastapi() 

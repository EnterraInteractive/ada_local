"""
Voice Assistant - Main orchestrator for Alexa-like voice interaction.
Manages: STT → Function Gemma → Qwen → TTS pipeline.
"""

import threading
import re
from typing import Optional
from PySide6.QtCore import QObject, Signal

from config import MAX_HISTORY, GRAY, RESET, CYAN, GREEN, WAKE_WORD
from core.stt import STTListener
from core.tts import tts, SentenceBuffer
from core.function_executor import executor as function_executor
from core.inference import runtime as inference_runtime
from core.command_router import route_command


class VoiceAssistant(QObject):
    """Main voice assistant orchestrator."""
    
    # Signals for UI updates (optional)
    wake_word_detected = Signal()
    speech_recognized = Signal(str)
    processing_started = Signal()
    processing_finished = Signal()
    error_occurred = Signal(str)
    # GUI update signals
    timer_set = Signal(int, str)  # seconds, label
    alarm_added = Signal()
    calendar_updated = Signal()
    task_added = Signal()
    
    def __init__(self):
        super().__init__()
        self.stt_listener: Optional[STTListener] = None
        self.running = False
        self.messages = [
            {
                'role': 'system', 
                'content': 'You are a helpful assistant. Respond in short, complete sentences. Never use emojis or special characters. Keep responses concise and conversational.'
            }
        ]
        self.current_session_id = None
        
    def initialize(self) -> bool:
        """Initialize voice assistant components."""
        try:
            print(f"{CYAN}[VoiceAssistant] Initializing voice assistant components...{RESET}")
            # Initialize STT listener
            print(f"{CYAN}[VoiceAssistant] Creating STT listener...{RESET}")
            self.stt_listener = STTListener(
                wake_word_callback=self._on_wake_word,
                speech_callback=self._on_speech
            )
            print(f"{CYAN}[VoiceAssistant] ✓ STT listener created{RESET}")
            
            print(f"{CYAN}[VoiceAssistant] Initializing STT models...{RESET}")
            if not self.stt_listener.initialize():
                print(f"{GRAY}[VoiceAssistant] ✗ Failed to initialize STT.{RESET}")
                return False
            print(f"{CYAN}[VoiceAssistant] ✓ STT initialized{RESET}")
            
            # Ensure TTS is initialized
            if not tts.piper_exe:
                print(f"{CYAN}[VoiceAssistant] Initializing TTS...{RESET}")
                tts.initialize()
                print(f"{CYAN}[VoiceAssistant] ✓ TTS initialized{RESET}")
            
            print(f"{CYAN}[VoiceAssistant] ✓ Voice assistant initialized successfully{RESET}")
            return True
        except Exception as e:
            print(f"{GRAY}[VoiceAssistant] ✗ Initialization error: {e}{RESET}")
            import traceback
            traceback.print_exc()
            return False
    
    def start(self):
        """Start the voice assistant."""
        if self.running:
            return
        
        if not self.stt_listener:
            if not self.initialize():
                return
        
        self.running = True
        self.stt_listener.start()
        print(f"{CYAN}[VoiceAssistant] Voice assistant started. Say '{GREEN}{WAKE_WORD}{RESET}{CYAN}' to activate.{RESET}")
    
    def stop(self):
        """Stop the voice assistant."""
        if not self.running:
            return
        
        self.running = False
        if self.stt_listener:
            listener = self.stt_listener
            self.stt_listener = None
            listener.stop()
            del listener
        print(f"{GRAY}[VoiceAssistant] Voice assistant stopped.{RESET}")
    
    def _on_wake_word(self):
        """Handle wake word detection."""
        print(f"{GREEN}[VoiceAssistant] ✓ Wake word callback received!{RESET}")
        print(f"{GREEN}[VoiceAssistant] Emitting wake_word_detected signal...{RESET}")
        self.wake_word_detected.emit()
        print(f"{GREEN}[VoiceAssistant] ✓ Signal emitted. Listening for speech...{RESET}")
    
    def _on_speech(self, text: str):
        """Handle recognized speech after wake word."""
        if not text.strip():
            return
        
        # Remove wake word from text if present
        text = text.lower().replace("ada", "").strip()
        if not text:
            return
        
        self.speech_recognized.emit(text)
        self.processing_started.emit()
        
        print(f"{CYAN}[VoiceAssistant] Processing: {text}{RESET}")
        
        # Process in background thread to avoid blocking
        thread = threading.Thread(
            target=self._process_query,
            args=(text,),
            daemon=True
        )
        thread.start()
    
    def _process_query(self, user_text: str):
        """Process recognized speech through the embedded K2 model."""
        try:
            func_name, params = route_command(user_text)
            context_prompt = user_text

            if func_name:
                result = function_executor.execute(func_name, params)
                if not result.get("success"):
                    context_prompt = (
                        f"The requested action could not be completed: "
                        f"{result.get('message', 'Unknown error')}. "
                        f"User request: {user_text}"
                    )
                else:
                    context_prompt = (
                        f"A.D.A. executed the requested action. "
                        f"Action: {func_name}. Result: {result.get('message', '')}. "
                        f"Structured result: {result.get('data')}. "
                        f"User request: {user_text}. "
                        "Confirm the result naturally and concisely."
                    )

                    if func_name == "set_timer":
                        data = result.get("data") or {}
                        self.timer_set.emit(
                            int(data.get("seconds", 0)),
                            data.get("label", "Timer"),
                        )
                    elif func_name == "set_alarm":
                        self.alarm_added.emit()
                    elif func_name == "create_calendar_event":
                        self.calendar_updated.emit()
                    elif func_name == "add_task":
                        self.task_added.emit()

            self._stream_local_response(context_prompt)

        except Exception as e:
            error_msg = f"Error processing voice request: {e}"
            print(f"{GRAY}[VoiceAssistant] {error_msg}{RESET}")
            self.error_occurred.emit(error_msg)
            self.processing_finished.emit()

    def _stream_local_response(self, user_text: str):
        """Generate and speak a response using A.D.A.'s embedded K2 model."""
        try:
            if len(self.messages) > MAX_HISTORY:
                self.messages = [self.messages[0]] + self.messages[-(MAX_HISTORY - 1):]

            self.messages.append({"role": "user", "content": user_text})
            sentence_buffer = SentenceBuffer()
            full_response = ""

            for chunk in inference_runtime.stream_chat(
                self.messages,
                max_tokens=512,
                temperature=0.4,
                top_p=0.9,
            ):
                full_response += chunk
                for sentence in sentence_buffer.add(chunk):
                    tts.queue_sentence(sentence)

            remaining = sentence_buffer.flush()
            if remaining:
                tts.queue_sentence(remaining)

            self.messages.append({"role": "assistant", "content": full_response})
            print(f"{GREEN}[VoiceAssistant] Response generated.{RESET}")
            self.processing_finished.emit()

        except Exception as e:
            print(f"{GRAY}[VoiceAssistant] Embedded response error: {e}{RESET}")
            self.error_occurred.emit(str(e))
            self.processing_finished.emit()


# Global voice assistant instance
voice_assistant = VoiceAssistant()

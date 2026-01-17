#!/usr/bin/env python3
"""
Delirium Detection System - FFT ONLY VERSION
Keyword detection disabled due to mic conflicts
Uses FFT audio analysis + facial emotion detection
"""

import cv2
import numpy as np
import pyaudio
import struct
from scipy.fft import fft
import time
import sys
from datetime import datetime
from deepface import DeepFace
import os
import subprocess

# Config
AUDIO_RATE = 16000
AUDIO_CHUNK = 1024
AUDIO_FORMAT = pyaudio.paInt16

VOICE_CONFIDENCE_THRESHOLD = 0.35
DISTRESS_ENERGY_THRESHOLD = 0.05
EMOTION_CONFIDENCE_THRESHOLD = 0.60

DISTRESS_EMOTIONS = ['angry', 'fear', 'sad', 'disgust']

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

AUDIO_CHECK_INTERVAL = 1.0
ALERT_DURATION = 5
SAVE_ALERT_IMAGES = True
VISION_ANALYSIS_FRAMES = 3

# Alert sound 
class AlertSound:
    """Plays alert sound through speaker"""
    
    def __init__(self):
        print("✓ Alert sound system ready")
    
    def play(self):
        """Play alert beep"""
        try:
            subprocess.run(['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1'], 
                          stdout=subprocess.DEVNULL, 
                          stderr=subprocess.DEVNULL,
                          timeout=2)
            print("🔊 BEEP")
        except:
            try:
                subprocess.run(['aplay', '/usr/share/sounds/alsa/Front_Center.wav'],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=2)
                print("🔊 BEEP")
            except:
                print("⚠️  Speaker not found")

# AUDIO ANALYZER
class AudioAnalyzer:
    """FFT audio analysis"""
    
    def __init__(self):
        self.p = pyaudio.PyAudio()
        
        print("\n=== Audio Input Devices ===")
        razer_index = None
        webcam_index = None
        default_index = None
        
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']}")
                
                if 'razer' in info['name'].lower() or 'seiren' in info['name'].lower():
                    razer_index = i
                    print(f"       ← RAZER MIC")
                
                if 'camera' in info['name'].lower() and 'usb' in info['name'].lower():
                    webcam_index = i
                    print(f"       ← WEBCAM MIC")
                
                if 'default' in info['name'].lower():
                    default_index = i
        
        # Priority: Razer > Webcam > Default
        if razer_index is not None:
            mic_index = razer_index
            print(f"\n→ Using RAZER MIC (index {mic_index})")
        elif webcam_index is not None:
            mic_index = webcam_index
            print(f"\n→ Using WEBCAM MIC (index {mic_index})")
        elif default_index is not None:
            mic_index = default_index
            print(f"\n→ Using DEFAULT MIC (index {mic_index})")
        else:
            mic_index = None
        
        # Open stream
        try:
            if mic_index is not None:
                self.stream = self.p.open(
                    format=AUDIO_FORMAT,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=AUDIO_CHUNK
                )
            else:
                self.stream = self.p.open(
                    format=AUDIO_FORMAT,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=AUDIO_CHUNK
                )
            print("✓ Audio FFT stream opened")
        except Exception as e:
            print(f"✗ Audio failed: {e}")
            try:
                print("→ Trying default...")
                self.stream = self.p.open(
                    format=AUDIO_FORMAT,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    frames_per_buffer=AUDIO_CHUNK
                )
                print("✓ Audio stream opened")
            except Exception as e2:
                print(f"✗ Complete failure: {e2}")
                sys.exit(1)
    
    def analyze_audio(self):
        try:
            data = self.stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            samples = np.array(struct.unpack(f'{AUDIO_CHUNK}h', data))
            samples = samples / 32768.0
            
            energy = np.mean(np.abs(samples))
            
            if energy < 0.005:
                return {
                    'voice_confidence': 0,
                    'is_distress': False,
                    'peak_frequency': 0,
                    'energy': energy
                }
            
            fft_data = fft(samples)
            fft_magnitude = np.abs(fft_data[:AUDIO_CHUNK//2])
            frequencies = np.fft.fftfreq(AUDIO_CHUNK, 1/AUDIO_RATE)[:AUDIO_CHUNK//2]
            
            peak_idx = np.argmax(fft_magnitude)
            peak_freq = abs(frequencies[peak_idx])
            
            fund_mask = (frequencies >= 85) & (frequencies <= 500)
            fund_energy = np.sum(fft_magnitude[fund_mask])
            
            harm_mask = (frequencies >= 500) & (frequencies <= 2000)
            harm_energy = np.sum(fft_magnitude[harm_mask])
            
            high_mask = (frequencies >= 2000) & (frequencies <= 5000)
            high_energy = np.sum(fft_magnitude[high_mask])
            
            total = np.sum(fft_magnitude) + 0.001
            fund_ratio = fund_energy / total
            harm_ratio = harm_energy / total
            high_ratio = high_energy / total
            
            voice_score = 0.0
            if fund_ratio > 0.05:
                voice_score += 0.30
            if harm_ratio > 0.10:
                voice_score += 0.25
            if 300 <= np.sum(frequencies * fft_magnitude) / total <= 4000:
                voice_score += 0.20
            
            is_distress = (
                (energy > DISTRESS_ENERGY_THRESHOLD and voice_score > VOICE_CONFIDENCE_THRESHOLD) or
                (energy > DISTRESS_ENERGY_THRESHOLD * 2.0) or
                (high_ratio > 0.15 and energy > DISTRESS_ENERGY_THRESHOLD * 0.8)
            )
            
            return {
                'voice_confidence': voice_score,
                'is_distress': is_distress,
                'peak_frequency': peak_freq,
                'energy': energy
            }
            
        except Exception as e:
            return {
                'voice_confidence': 0,
                'is_distress': False,
                'peak_frequency': 0,
                'energy': 0
            }
    
    def close(self):
        try:
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, 'p'):
                self.p.terminate()
        except:
            pass

# EMOTION ANALYZER
class EmotionAnalyzer:
    """DeepFace facial emotion detection"""
    
    def __init__(self):
        print("\n=== Emotion Detection ===")
        print("✓ DeepFace ready")
        self.cap = None
        self.window_created = False
    
    def open_camera(self):
        print("→ Opening camera...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("✗ Camera failed!")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        print("→ Flushing buffer...")
        for _ in range(10):
            self.cap.read()
            time.sleep(0.05)
        
        if not self.window_created:
            cv2.namedWindow('Delirium Detection', cv2.WINDOW_NORMAL)
            self.window_created = True
        
        print("✓ Camera ready")
        return True
    
    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("→ Camera closed")
    
    def analyze_current_frame(self):
        if self.cap is None:
            return {'is_distressed': False, 'emotion': None, 'confidence': 0, 'frame': None}
        
        ret, frame = self.cap.read()
        if not ret:
            return {'is_distressed': False, 'emotion': None, 'confidence': 0, 'frame': None}
        
        try:
            result = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion] / 100.0
            
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            is_distressed = dominant_emotion in DISTRESS_EMOTIONS and confidence > EMOTION_CONFIDENCE_THRESHOLD
            
            color = (0, 0, 255) if is_distressed else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            label = f"{dominant_emotion.upper()}: {confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y-lh-10), (x+lw, y), color, -1)
            cv2.putText(frame, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show emotions
            y_offset = 30
            for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                text = f"{emotion}: {score:.1f}%"
                color_text = (0, 0, 255) if emotion in DISTRESS_EMOTIONS else (0, 255, 0)
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)
                y_offset += 20
            
            cv2.imshow('Delirium Detection', frame)
            cv2.waitKey(1)
            
            return {
                'is_distressed': is_distressed,
                'emotion': dominant_emotion,
                'confidence': confidence,
                'frame': frame
            }
            
        except Exception as e:
            cv2.putText(frame, "No face", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow('Delirium Detection', frame)
            cv2.waitKey(1)
            
            return {'is_distressed': False, 'emotion': None, 'confidence': 0, 'frame': frame}
    
    def close(self):
        self.close_camera()
        cv2.destroyAllWindows()

#MAIN SYSTEM
class DeliriumDetector:
    """Main orchestrator"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("  DELIRIUM DETECTION SYSTEM")
        print("  FFT Audio Analysis + Facial Emotion Detection")
        print("="*60 + "\n")
        
        self.audio = AudioAnalyzer()
        self.emotion = EmotionAnalyzer()
        self.alert_sound = AlertSound()
        
        self.state = "LISTENING"
        self.last_audio_check = time.time()
        self.alert_count = 0
        
        if not os.path.exists('alerts'):
            os.makedirs('alerts')
        
        print("\n" + "="*60)
        print("  ✓ SYSTEM READY")
        print("  Listening for distress sounds...")
        print("  Press Ctrl+C to stop")
        print("="*60 + "\n")
    
    def run(self):
        try:
            while True:
                current_time = time.time()
                
                if self.state == "LISTENING":
                    if current_time - self.last_audio_check >= AUDIO_CHECK_INTERVAL:
                        self.last_audio_check = current_time
                        audio_result = self.audio.analyze_audio()
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              f"V:{audio_result['voice_confidence']:.2f} | "
                              f"{audio_result['peak_frequency']:.0f}Hz | "
                              f"E:{audio_result['energy']:.3f} | "
                              f"{'YES' if audio_result['is_distress'] else 'NO'}")
                        
                        if audio_result['is_distress']:
                            print(f"\n{'!'*60}")
                            print("  🚨 DISTRESS DETECTED")
                            print("  → ACTIVATING CAMERA")
                            print(f"{'!'*60}\n")
                            self.state = "ANALYZING"
                
                elif self.state == "ANALYZING":
                    if not self.emotion.open_camera():
                        print("✗ Camera failed")
                        self.state = "LISTENING"
                        continue
                    
                    print(f"→ Analyzing {VISION_ANALYSIS_FRAMES} frames...")
                    distress_detections = []
                    
                    for i in range(VISION_ANALYSIS_FRAMES):
                        time.sleep(0.2)
                        result = self.emotion.analyze_current_frame()
                        
                        if result['emotion']:
                            status = "✓ DISTRESS" if result['is_distressed'] else "- normal"
                            print(f"  F{i+1}: {result['emotion']:8s} {result['confidence']:.2f} {status}")
                            distress_detections.append(result)
                        else:
                            print(f"  F{i+1}: No face")
                    
                    self.emotion.close_camera()
                    
                    distressed = [r for r in distress_detections if r['is_distressed']]
                    
                    if distressed:
                        best = max(distressed, key=lambda x: x['confidence'])
                        print(f"\n→ Distress in {len(distressed)}/{len(distress_detections)} frames")
                        self.trigger_alert(best)
                    else:
                        print(f"\n→ No distress detected")
                        print("→ Returning to monitoring\n")
                    
                    self.state = "LISTENING"
                    time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("  Shutting down...")
            print("="*60)
            self.cleanup()
    
    def trigger_alert(self, result):
        """Trigger alert"""
        self.alert_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*60)
        print("  🚨🚨🚨 ALERT TRIGGERED 🚨🚨🚨")
        print("  PATIENT DISTRESS CONFIRMED")
        print(f"  Emotion: {result['emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Alert #{self.alert_count}")
        print(f"  Time: {timestamp}")
        print("  → CALL NURSE IMMEDIATELY")
        print("="*60 + "\n")
        
        # PLAY ALERT
        self.alert_sound.play()
        
        # Save image
        if result['frame'] is not None:
            filename = f"alerts/alert_{self.alert_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, result['frame'])
            print(f"✓ Saved: {filename}\n")
        
        time.sleep(ALERT_DURATION)
        print("→ Alert acknowledged, resuming\n")
    
    def cleanup(self):
        """Clean shutdown"""
        self.audio.close()
        self.emotion.close()
        print("✓ Stopped\n")

# ENTRY POINT
if __name__ == "__main__":
    try:
        detector = DeliriumDetector()
        detector.run()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


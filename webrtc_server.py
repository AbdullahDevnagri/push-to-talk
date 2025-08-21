import os
import asyncio
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIClientMessageFrame,
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams

load_dotenv("server/.env", override=True)


class PushToTalkGate(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._gate_opened = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)

        elif isinstance(frame, RTVIClientMessageFrame):
            self._handle_rtvi_frame(frame)
            await self.push_frame(frame, direction)

        # If the gate is closed, suppress all audio frames until the user releases the button
        if not self._gate_opened and isinstance(
            frame,
            (
                InputAudioRawFrame,
                UserStartedSpeakingFrame,
                StartInterruptionFrame,
                StopInterruptionFrame,
            ),
        ):
            logger.trace(f"{frame.__class__.__name__} suppressed - Button not pressed")
        else:
            await self.push_frame(frame, direction)

    def _handle_rtvi_frame(self, frame: RTVIClientMessageFrame):
        if frame.type == "push_to_talk" and frame.data:
            data = frame.data
            if data.get("state") == "start":
                self._gate_opened = True
                logger.info("Input gate opened - user started talking")
            elif data.get("state") == "stop":
                self._gate_opened = False
                logger.info("Input gate closed - user stopped talking")


async def run_webrtc_bot():
    logger.info("Starting WebRTC bot server...")
    
    # Create WebRTC transport
    webrtc_transport = SmallWebRTCTransport(
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Use the same services as your original bot
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )
    
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Push-to-talk gate for controlling audio input
    push_to_talk_gate = PushToTalkGate()

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            webrtc_transport.input(),  # Transport user input
            rtvi,
            push_to_talk_gate,
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            webrtc_transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @webrtc_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("WebRTC Client connected")
        # Send StartFrame to initialize the pipeline properly
        await task.queue_frames([StartFrame()])

    @webrtc_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("WebRTC Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    logger.info("WebRTC server starting...")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(run_webrtc_bot())

'use client';

import { FullScreenContainer, ThemeProvider } from '@pipecat-ai/voice-ui-kit';
import { App } from './components/App';

export default function Home() {
  return (
    <ThemeProvider>
      <FullScreenContainer>
        <App
          transportType="webrtc"
          connectParams={{
            webrtcUrl: 'http://localhost:7860/api/offer',
          }}
        />
      </FullScreenContainer>
    </ThemeProvider>
  );
}

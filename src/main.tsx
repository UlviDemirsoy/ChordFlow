import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import {
  DatasetCollector,
  RightDatasetCollector,
} from './components/collector/DatasetCollector.tsx'
import { ClassifierTest } from './components/classifier/ClassifierTest.tsx'
import { RightClassifierTest } from './components/classifier/RightClassifierTest.tsx'
import { InstrumentView } from './components/instrument/InstrumentView.tsx'

const currentPath = window.location.pathname
if (currentPath === '/') {
  window.location.replace('/play')
}

const page =
  currentPath === '/collect' ? (
    <DatasetCollector />
  ) : currentPath === '/collect/right' ? (
    <RightDatasetCollector />
  ) : currentPath === '/classify' ? (
    <ClassifierTest />
  ) : currentPath === '/classify/right' ? (
    <RightClassifierTest />
  ) : currentPath === '/play' || currentPath === '/' ? (
    <InstrumentView />
  ) : (
    <App />
  )

createRoot(document.getElementById('root')!).render(
  <StrictMode>{page}</StrictMode>,
)

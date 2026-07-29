import { randomUUID } from 'node:crypto'
import {
  access,
  appendFile,
  mkdir,
  readdir,
  unlink,
  writeFile,
} from 'node:fs/promises'
import path from 'node:path'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

interface LandmarkPayload {
  x: number
  y: number
  z: number
}

interface CapturePayload {
  degree: number
  imageDataUrl: string
  capturedAt: string
  handednessScore: number
  landmarks: LandmarkPayload[]
}

type HandName = 'left' | 'right'

const DATASET_CONFIG: Record<HandName, readonly number[]> = {
  left: [0, 1, 2, 3, 4, 5, 6, 7],
  right: [0, 1, 2, 3, 4, 5],
}
const MAX_BODY_BYTES = 15 * 1024 * 1024

function csvHeader() {
  const landmarkColumns = Array.from({ length: 21 }, (_, index) => [
    `x${index}`,
    `y${index}`,
    `z${index}`,
  ]).flat()

  return [
    'id',
    'label',
    'image_path',
    'captured_at',
    'handedness_score',
    ...landmarkColumns,
  ].join(',')
}

function sendJson(
  response: ServerResponse,
  statusCode: number,
  payload: unknown,
) {
  response.statusCode = statusCode
  response.setHeader('Content-Type', 'application/json; charset=utf-8')
  response.end(JSON.stringify(payload))
}

async function readBody(request: IncomingMessage) {
  return new Promise<string>((resolve, reject) => {
    let body = ''
    let size = 0

    request.setEncoding('utf8')
    request.on('data', (chunk: string) => {
      size += Buffer.byteLength(chunk)
      if (size > MAX_BODY_BYTES) {
        reject(new Error('İstek gövdesi çok büyük.'))
        request.destroy()
        return
      }
      body += chunk
    })
    request.on('end', () => resolve(body))
    request.on('error', reject)
  })
}

function validatePayload(
  value: unknown,
  hand: HandName,
  classLabels: readonly number[],
): CapturePayload {
  if (!value || typeof value !== 'object') {
    throw new Error('Geçersiz kayıt verisi.')
  }

  const payload = value as Partial<CapturePayload>
  if (
    !Number.isInteger(payload.degree) ||
    !classLabels.includes(payload.degree as number)
  ) {
    throw new Error(
      `Sınıf etiketi ${classLabels[0]}–${classLabels.at(-1)} arasında olmalı.`,
    )
  }

  if (
    typeof payload.imageDataUrl !== 'string' ||
    !payload.imageDataUrl.startsWith('data:image/jpeg;base64,')
  ) {
    throw new Error('JPEG görüntüsü bulunamadı.')
  }

  if (!Array.isArray(payload.landmarks) || payload.landmarks.length !== 21) {
    throw new Error(
      `${hand === 'left' ? 'Sol' : 'Sağ'} ele ait 21 landmark gerekli.`,
    )
  }

  for (const landmark of payload.landmarks) {
    if (
      !Number.isFinite(landmark.x) ||
      !Number.isFinite(landmark.y) ||
      !Number.isFinite(landmark.z)
    ) {
      throw new Error('Landmark koordinatları geçersiz.')
    }
  }

  if (
    typeof payload.capturedAt !== 'string' ||
    !Number.isFinite(payload.handednessScore)
  ) {
    throw new Error('Kayıt metadatası eksik.')
  }

  return payload as CapturePayload
}

function createDatasetHandler(
  hand: HandName,
  classLabels: readonly number[],
): {
  ensureStorage: () => Promise<void>
  handler: (request: IncomingMessage, response: ServerResponse) => Promise<void>
} {
  const datasetRoot = path.resolve(process.cwd(), 'model', 'data', hand)
  const csvPath = path.join(datasetRoot, 'landmarks.csv')
  let writeQueue = Promise.resolve()

  const ensureStorage = async () => {
    await Promise.all(
      classLabels.map((label) =>
        mkdir(path.join(datasetRoot, label.toString()), { recursive: true }),
      ),
    )

    try {
      await access(csvPath)
    } catch {
      await writeFile(csvPath, `${csvHeader()}\n`, 'utf8')
    }
  }

  const getCounts = async () => {
    await ensureStorage()
    const entries = await Promise.all(
      classLabels.map(async (label) => {
        const files = await readdir(path.join(datasetRoot, label.toString()))
        return [label, files.filter((file) => file.endsWith('.jpg')).length]
      }),
    )
    return Object.fromEntries(entries) as Record<string, number>
  }

  const handler = async (
    request: IncomingMessage,
    response: ServerResponse,
  ) => {
    try {
      if (request.method === 'GET') {
        const counts = await getCounts()
        sendJson(response, 200, {
          counts,
          total: Object.values(counts).reduce(
            (sum, count) => sum + count,
            0,
          ),
        })
        return
      }

      if (request.method !== 'POST') {
        sendJson(response, 405, { error: 'Desteklenmeyen HTTP metodu.' })
        return
      }

      const payload = validatePayload(
        JSON.parse(await readBody(request)),
        hand,
        classLabels,
      )
      const id = randomUUID()
      const relativeImagePath = `${payload.degree}/${id}.jpg`
      const imagePath = path.join(datasetRoot, relativeImagePath)
      const imageBuffer = Buffer.from(
        payload.imageDataUrl.slice('data:image/jpeg;base64,'.length),
        'base64',
      )
      const landmarkValues = payload.landmarks.flatMap((landmark) => [
        landmark.x,
        landmark.y,
        landmark.z,
      ])
      const csvRow = [
        id,
        payload.degree,
        relativeImagePath,
        payload.capturedAt,
        payload.handednessScore,
        ...landmarkValues,
      ].join(',')

      const writeOperation = writeQueue.then(async () => {
        await writeFile(imagePath, imageBuffer)
        try {
          await appendFile(csvPath, `${csvRow}\n`, 'utf8')
        } catch (error) {
          await unlink(imagePath).catch(() => undefined)
          throw error
        }
      })
      writeQueue = writeOperation.catch(() => undefined)
      await writeOperation

      sendJson(response, 201, {
        id,
        imagePath: `model/data/${hand}/${relativeImagePath}`,
      })
    } catch (error) {
      const message =
        error instanceof Error ? error.message : 'Dataset kaydı başarısız.'
      sendJson(response, 400, { error: message })
    }
  }

  return { ensureStorage, handler }
}

export function datasetWriterPlugin(): Plugin {
  const stores = Object.entries(DATASET_CONFIG).map(([hand, classLabels]) => ({
    hand: hand as HandName,
    ...createDatasetHandler(hand as HandName, classLabels),
  }))

  return {
    name: 'chordflow-dataset-writer',
    configureServer(server) {
      stores.forEach(({ hand, ensureStorage, handler }) => {
        void ensureStorage()
        server.middlewares.use(`/api/dataset/${hand}`, handler)
      })
    },
  }
}

import { isModelId, type ModelTask, type OpenRouterCatalog, type OpenRouterModel } from '../shared/openrouter-models'
import { readResponseText } from './http-response'

const CACHE_MS = 10 * 60 * 1000
let cached: OpenRouterCatalog | null = null
let pending: Promise<OpenRouterCatalog> | null = null

const OPENCODE_GO_BASE = (process.env.OPENCODE_BASE_URL || 'https://opencode.ai/zen/go/v1').replace(/\/$/, '')

// Static OpenCode Go catalog: Muse Spark Contributor (Responses API) + Go chat models.
// Live fetch augments this; static entries guarantee the app works offline.
const STATIC_PLANNING: OpenRouterModel[] = [
  { id: 'muse-spark-1.3-contributor', name: 'Muse Spark 1.3 Contributor', contextLength: 1048576, maxOutputTokens: 131072, supportsImages: true, inputPrice: 0.1, outputPrice: 0.2, unavailableReason: null },
  { id: 'muse-spark-1.2-contributor', name: 'Muse Spark 1.2 Contributor', contextLength: 1048576, maxOutputTokens: 65536, supportsImages: true, inputPrice: 0.1, outputPrice: 0.2, unavailableReason: null },
  { id: 'kimi-k3', name: 'Kimi K3', contextLength: 262144, maxOutputTokens: 32000, supportsImages: true, inputPrice: 3.0, outputPrice: 15.0, unavailableReason: null },
  { id: 'glm-5.2', name: 'GLM 5.2', contextLength: 262144, maxOutputTokens: 32000, supportsImages: true, inputPrice: 1.4, outputPrice: 4.4, unavailableReason: null },
  { id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro', contextLength: 262144, maxOutputTokens: 32000, supportsImages: false, inputPrice: 1.32, outputPrice: 3.96, unavailableReason: null },
]

const STATIC_TRANSCRIPTION: OpenRouterModel[] = [
  { id: 'parakeet-local', name: 'Parakeet Local (offline, no key)', contextLength: null, maxOutputTokens: null, supportsImages: false, inputPrice: 0, outputPrice: 0, unavailableReason: null },
]

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
}

function number(value: unknown, positive = false): number | null {
  if (typeof value !== 'number' && typeof value !== 'string') return null
  if (value === '') return null
  const n = Number(value)
  return Number.isFinite(n) && (positive ? n > 0 : n >= 0) ? n : null
}

export function parseModelCatalog(value: unknown, task: ModelTask): OpenRouterModel[] {
  // OpenCode Go /models returns [{id}] without architecture/pricing.
  const data = record(value).data
  if (!Array.isArray(data) || data.length > 10000) throw new Error('OpenCode Go returned an invalid model catalog.')
  const models = new Map<string, OpenRouterModel>()
  const statics = task === 'planning' ? STATIC_PLANNING : STATIC_TRANSCRIPTION
  for (const s of statics) models.set(s.id, s)
  for (const entry of data) {
    const raw = record(entry)
    const id = typeof raw.id === 'string' ? raw.id : ''
    if (!id) continue
    // Planning: Muse + Go chat models. Transcription: local only (Go has no STT).
    if (task === 'transcription') continue
    if (models.has(id)) continue
    // Accept Go IDs (no slash) plus legacy slash IDs.
    if (!/^[a-z0-9][a-z0-9._:-]*$/i.test(id) && !isModelId(id)) continue
    models.set(id, {
      id,
      name: typeof raw.name === 'string' ? raw.name.slice(0, 160) : id,
      contextLength: null,
      maxOutputTokens: null,
      supportsImages: id.toLowerCase().includes('muse') || id.toLowerCase().includes('kimi') || id.toLowerCase().includes('glm'),
      inputPrice: null,
      outputPrice: null,
      unavailableReason: null
    })
  }
  if (!models.size) throw new Error('OpenCode Go returned an empty model catalog. Try refreshing.')
  return [...models.values()].sort((a, b) => Number(Boolean(a.unavailableReason)) - Number(Boolean(b.unavailableReason)) || a.name.localeCompare(b.name))
}

async function fetchModels(task: ModelTask): Promise<OpenRouterModel[]> {
  try {
    // Public read-only catalog. No API key leaves the main process.
    const response = await fetch(`${OPENCODE_GO_BASE}/models`, {
      redirect: 'error', signal: AbortSignal.timeout(15000), headers: { Accept: 'application/json' }
    })
    if (!response.ok) {
      await response.body?.cancel()
      throw new Error('Model catalog unavailable')
    }
    return parseModelCatalog(JSON.parse(await readResponseText(response, 8 * 1024 * 1024)), task)
  } catch {
    // Offline fallback: static catalog so Advanced pickers still work.
    return task === 'planning' ? STATIC_PLANNING : STATIC_TRANSCRIPTION
  }
}

export async function getModelCatalog(refresh: unknown = false): Promise<OpenRouterCatalog> {
  if (typeof refresh !== 'boolean') throw new Error('Invalid model refresh option')
  if (pending) return pending
  if (!refresh && cached && Date.now() - Date.parse(cached.fetchedAt) < CACHE_MS) return cached
  pending = Promise.all([fetchModels('planning'), fetchModels('transcription')]).then(([planning, transcription]) => {
    cached = { planning, transcription, fetchedAt: new Date().toISOString() }
    return cached
  })
  try { return await pending } finally { pending = null }
}

export async function resolveAdvancedModels(plannerId: string, transcriptionId: string): Promise<OpenRouterModel> {
  const catalog = await getModelCatalog()
  for (const [task, id] of [['planning', plannerId], ['transcription', transcriptionId]] as const) {
    const model = catalog[task].find((item) => item.id === id)
    if (!model) throw new Error(`The selected ${task} model is no longer listed. Refresh the models in Advanced mode.`)
    if (model.unavailableReason) throw new Error(model.unavailableReason)
  }
  return catalog.planning.find((item) => item.id === plannerId)!
}

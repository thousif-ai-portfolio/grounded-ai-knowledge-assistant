import type {
  Answer,
  DocumentInfo,
  EvaluationResults,
  HistoryItem,
  SearchMode,
  UploadResult,
  User,
} from './types'

const TOKEN_KEY = 'gaka_token'

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    // Storage can be unavailable (private mode); the session then lasts until reload.
  }
}

async function send(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`/api${path}`, { ...init, headers })
  if (!response.ok) {
    let message = response.statusText || 'Request failed'
    try {
      const body = await response.json()
      // FastAPI sends either a string or a list of validation errors.
      message = typeof body.detail === 'string' ? body.detail : (body.detail?.[0]?.msg ?? message)
    } catch {
      // Not JSON; keep the status text.
    }
    throw new ApiError(response.status, message)
  }
  return response
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await send(path, init)
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
}

type StreamEvent =
  | { type: 'delta'; text: string }
  | ({ type: 'answer' } & Answer)
  | { type: 'error'; detail: string }

/** Calls `onText` with each piece of the answer as the model writes it; resolves with the final answer. */
async function askStream(
  question: string,
  mode: SearchMode,
  onText: (text: string) => void,
): Promise<Answer> {
  const response = await send('/ask/stream', json({ question, mode }))
  if (!response.body) throw new ApiError(0, 'The answer stream was empty')
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (value) buffer += value
    // One JSON event per line; the last line may still be incomplete.
    const lines = buffer.split('\n')
    buffer = done ? '' : (lines.pop() ?? '')
    for (const line of lines) {
      if (!line.trim()) continue
      const event = JSON.parse(line) as StreamEvent
      if (event.type === 'delta') onText(event.text)
      else if (event.type === 'error') throw new ApiError(503, event.detail)
      else {
        const { type: _, ...answer } = event
        return answer
      }
    }
    if (done) throw new ApiError(0, 'The answer stream ended early')
  }
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  register: (username: string, password: string) =>
    request<User>('/auth/register', json({ username, password })),

  async login(username: string, password: string): Promise<void> {
    // OAuth2 password flow expects form fields, not JSON.
    const { access_token } = await request<{ access_token: string }>('/auth/login', {
      method: 'POST',
      body: new URLSearchParams({ username, password }),
    })
    setToken(access_token)
  },

  me: () => request<User>('/auth/me'),

  listDocuments: () => request<DocumentInfo[]>('/documents'),

  uploadDocument(file: File) {
    const form = new FormData()
    form.append('file', file)
    return request<UploadResult>('/documents', { method: 'POST', body: form })
  },

  deleteDocument: (docId: string) =>
    request<void>(`/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' }),

  askStream,

  history: () => request<HistoryItem[]>('/history'),

  clearHistory: () => request<void>('/history', { method: 'DELETE' }),

  evaluation: () => request<EvaluationResults>('/evaluation'),
}

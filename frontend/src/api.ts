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

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
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
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T)
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

  ask: (question: string, mode: SearchMode) => request<Answer>('/ask', json({ question, mode })),

  history: () => request<HistoryItem[]>('/history'),

  clearHistory: () => request<void>('/history', { method: 'DELETE' }),

  evaluation: () => request<EvaluationResults>('/evaluation'),
}

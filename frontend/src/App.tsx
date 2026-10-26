import { Suspense, lazy, useCallback, useEffect, useState } from 'react'
import { ApiError, api, getToken, setToken } from './api'
import { AuthForm } from './components/AuthForm'
import { ChatPage } from './components/ChatPage'
import { DocumentsPage } from './components/DocumentsPage'
import type { User } from './types'

// The dashboard pulls in the charting library, so it's only downloaded when opened.
const EvaluationPage = lazy(() =>
  import('./components/EvaluationPage').then((m) => ({ default: m.EvaluationPage })),
)

type Tab = 'chat' | 'documents' | 'evaluation'

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'documents', label: 'Documents' },
  { id: 'evaluation', label: 'Evaluation' },
]

export default function App() {
  const [user, setUser] = useState<User | null>(null)
  const [checking, setChecking] = useState(() => getToken() !== null)
  const [tab, setTab] = useState<Tab>('chat')

  useEffect(() => {
    if (!getToken()) return
    api
      .me()
      .then(setUser)
      .catch(() => setToken(null))
      .finally(() => setChecking(false))
  }, [])

  const signOut = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  // Turns any error into a message; an expired session sends the user back to sign in.
  const handleError = useCallback(
    (err: unknown) => {
      if (err instanceof ApiError && err.status === 401) signOut()
      return err instanceof Error ? err.message : 'Something went wrong'
    },
    [signOut],
  )

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900">
        <div className="mx-auto flex max-w-4xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <h1 className="flex items-center gap-2 font-semibold">
            <img src="/favicon.svg" alt="" className="h-6 w-6" />
            Grounded AI Knowledge Assistant
          </h1>
          <nav className="flex gap-1" aria-label="Main">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                aria-current={tab === t.id ? 'page' : undefined}
                className={`rounded-lg px-3 py-1.5 text-sm ${
                  tab === t.id
                    ? 'bg-indigo-50 font-medium text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          {user && (
            <div className="ml-auto flex items-center gap-3 text-sm">
              <span className="text-slate-500">{user.username}</span>
              <button className="text-indigo-600 hover:underline dark:text-indigo-400" onClick={signOut}>
                Sign out
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-4 py-6">
        {tab === 'evaluation' ? (
          <Suspense fallback={<p className="text-sm text-slate-500">Loading results…</p>}>
            <EvaluationPage />
          </Suspense>
        ) : checking ? (
          <p className="text-sm text-slate-500">Loading…</p>
        ) : !user ? (
          <AuthForm onSignedIn={setUser} />
        ) : tab === 'chat' ? (
          <ChatPage onError={handleError} />
        ) : (
          <DocumentsPage onError={handleError} />
        )}
      </main>
    </div>
  )
}

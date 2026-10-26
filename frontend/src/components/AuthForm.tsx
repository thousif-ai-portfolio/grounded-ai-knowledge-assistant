import { type FormEvent, useState } from 'react'
import { api } from '../api'
import type { User } from '../types'
import { inputClass, primaryButtonClass } from '../styles'
import { Card, ErrorNote } from './ui'

export function AuthForm({ onSignedIn }: { onSignedIn: (user: User) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      if (mode === 'register') await api.register(username, password)
      await api.login(username, password)
      onSignedIn(await api.me())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setBusy(false)
    }
  }

  const isLogin = mode === 'login'
  return (
    <Card className="mx-auto mt-10 max-w-sm">
      <h2 className="text-lg font-semibold">{isLogin ? 'Sign in' : 'Create an account'}</h2>
      <p className="mt-1 text-sm text-slate-500">
        Ask questions about your documents and get answers with sources.
      </p>
      <form onSubmit={submit} className="mt-5 space-y-3">
        <label className="block text-sm">
          Username
          <input
            className={`${inputClass} mt-1`}
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
        </label>
        <label className="block text-sm">
          Password
          <input
            className={`${inputClass} mt-1`}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={isLogin ? 'current-password' : 'new-password'}
            minLength={isLogin ? undefined : 8}
            required
          />
        </label>
        {error && <ErrorNote message={error} />}
        <button type="submit" className={`${primaryButtonClass} w-full`} disabled={busy}>
          {busy ? 'Please wait…' : isLogin ? 'Sign in' : 'Create account'}
        </button>
      </form>
      <button
        type="button"
        className="mt-4 text-sm text-indigo-600 hover:underline dark:text-indigo-400"
        onClick={() => {
          setMode(isLogin ? 'register' : 'login')
          setError('')
        }}
      >
        {isLogin ? 'New here? Create an account' : 'Already have an account? Sign in'}
      </button>
    </Card>
  )
}

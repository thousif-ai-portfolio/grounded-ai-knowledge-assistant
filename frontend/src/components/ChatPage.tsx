import { type FormEvent, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { SearchMode } from '../types'
import { MODE_LABELS } from '../types'
import { type AnswerView, AnswerCard } from './AnswerCard'
import { inputClass, primaryButtonClass, secondaryButtonClass } from '../styles'
import { ErrorNote } from './ui'

interface Turn {
  key: string
  question: string
  answer?: AnswerView
  // The answer so far, while the model is still writing it.
  draft?: string
  error?: string
}

const EXAMPLES = [
  'What does error E1042 mean?',
  'Why do neighbouring chunks share some repeated text?',
  'What is the capital of France?',
]

export function ChatPage({ onError }: { onError: (err: unknown) => string }) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState<SearchMode>('hybrid_rerank')
  const [busy, setBusy] = useState(false)
  const [loadError, setLoadError] = useState('')
  const bottom = useRef<HTMLDivElement>(null)
  const nextKey = useRef(0)

  useEffect(() => {
    api
      .history()
      .then((items) =>
        setTurns(
          items.reverse().map((item) => ({ key: `h${item.id}`, question: item.question, answer: item })),
        ),
      )
      .catch((err) => setLoadError(onError(err)))
  }, [onError])

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: 'smooth' })
  }, [turns])

  async function ask(text: string) {
    const trimmed = text.trim()
    if (!trimmed || busy) return
    const key = `t${++nextKey.current}`
    setTurns((prev) => [...prev, { key, question: trimmed }])
    setQuestion('')
    setBusy(true)
    try {
      const answer = await api.askStream(trimmed, mode, (text) =>
        setTurns((prev) => prev.map((t) => (t.key === key ? { ...t, draft: (t.draft ?? '') + text } : t))),
      )
      setTurns((prev) => prev.map((t) => (t.key === key ? { ...t, answer, draft: undefined } : t)))
    } catch (err) {
      const error = onError(err)
      setTurns((prev) => prev.map((t) => (t.key === key ? { ...t, error } : t)))
    } finally {
      setBusy(false)
    }
  }

  async function clear() {
    try {
      await api.clearHistory()
      setTurns([])
    } catch (err) {
      setLoadError(onError(err))
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {loadError && <ErrorNote message={loadError} />}
      {turns.length === 0 ? (
        <div className="rounded-xl border border-dashed border-slate-300 p-6 text-center text-sm text-slate-500 dark:border-slate-700">
          <p>Ask anything about the documents in the knowledge base. Try:</p>
          <div className="mt-3 flex flex-wrap justify-center gap-2">
            {EXAMPLES.map((example) => (
              <button key={example} className={secondaryButtonClass} onClick={() => ask(example)}>
                {example}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="flex justify-end">
          <button className={secondaryButtonClass} onClick={clear}>
            Clear history
          </button>
        </div>
      )}

      {turns.map((turn) => (
        <div key={turn.key} className="space-y-2">
          <div className="ml-auto w-fit max-w-[85%] rounded-xl bg-indigo-600 px-4 py-2 text-sm text-white">
            {turn.question}
          </div>
          {turn.answer && <AnswerCard answer={turn.answer} />}
          {turn.error && <ErrorNote message={turn.error} />}
          {!turn.answer &&
            !turn.error &&
            (turn.draft ? (
              <div className="rounded-xl border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900">
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {turn.draft}
                  <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-slate-400 align-text-bottom" />
                </p>
              </div>
            ) : (
              <p className="text-sm text-slate-500" aria-live="polite">
                Searching the documents and reading the best matches… on a CPU this can take a minute.
              </p>
            ))}
        </div>
      ))}
      <div ref={bottom} />

      <form
        onSubmit={(e: FormEvent) => {
          e.preventDefault()
          ask(question)
        }}
        className="sticky bottom-0 flex flex-col gap-2 border-t border-slate-200 bg-slate-50 py-3 sm:flex-row dark:border-slate-800 dark:bg-slate-950"
      >
        <input
          className={inputClass}
          placeholder="Ask a question…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          aria-label="Question"
          maxLength={2000}
        />
        <select
          className={`${inputClass} sm:w-44`}
          value={mode}
          onChange={(e) => setMode(e.target.value as SearchMode)}
          aria-label="Search mode"
        >
          {(Object.keys(MODE_LABELS) as SearchMode[]).map((m) => (
            <option key={m} value={m}>
              {MODE_LABELS[m]}
            </option>
          ))}
        </select>
        <button type="submit" className={primaryButtonClass} disabled={busy || !question.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}

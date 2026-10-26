import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AnswerCard } from '../components/AnswerCard'
import { AuthForm } from '../components/AuthForm'
import { EvaluationView } from '../components/EvaluationPage'
import type { EvaluationResults } from '../types'

const citation = {
  number: 1,
  filename: 'atlas-sync-manual.pdf',
  page: 4,
  text: 'E1042: the local cache is full.',
  score: 0.91,
}

describe('AnswerCard', () => {
  it('shows the answer with expandable sources', async () => {
    render(
      <AnswerCard
        answer={{
          answer: 'The local cache is full [1].',
          citations: [citation],
          grounded: true,
          mode: 'hybrid_rerank',
          retrieval_ms: 120,
          generation_ms: 900,
        }}
      />,
    )

    expect(screen.getByText('The local cache is full [1].')).toBeInTheDocument()
    expect(screen.getByText('atlas-sync-manual.pdf')).toBeInTheDocument()
    expect(screen.getByText('page 4')).toBeInTheDocument()
    expect(screen.getByText(/search 120 ms/)).toBeInTheDocument()
    expect(screen.queryByText(citation.text)).not.toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /\[1\]/ }))

    expect(screen.getByText(citation.text)).toBeInTheDocument()
  })

  it('explains an "I don\'t know" answer', () => {
    render(
      <AnswerCard answer={{ answer: "I don't know.", citations: [], grounded: false, mode: 'semantic' }} />,
    )

    expect(screen.getByText(/documents don't contain the answer/)).toBeInTheDocument()
    expect(screen.queryByText(/did not cite any source/)).not.toBeInTheDocument()
  })

  it('warns when an answer cites nothing', () => {
    render(<AnswerCard answer={{ answer: 'Maybe.', citations: [], grounded: false, mode: 'hybrid' }} />)

    expect(screen.getByText(/did not cite any source/)).toBeInTheDocument()
  })
})

describe('AuthForm', () => {
  it('registers, signs in and reports the user', async () => {
    const responses = [
      { id: 1, username: 'alice' },
      { access_token: 't', token_type: 'bearer' },
      { id: 1, username: 'alice' },
    ]
    const fetchMock = vi.fn(
      async (_url: string) => new Response(JSON.stringify(responses.shift()), { status: 200 }),
    )
    vi.stubGlobal('fetch', fetchMock)
    const onSignedIn = vi.fn()
    render(<AuthForm onSignedIn={onSignedIn} />)

    await userEvent.click(screen.getByText(/Create an account/))
    await userEvent.type(screen.getByLabelText('Username'), 'alice')
    await userEvent.type(screen.getByLabelText('Password'), 'long-password')
    await userEvent.click(screen.getByRole('button', { name: 'Create account' }))

    expect(fetchMock.mock.calls.map((c) => c[0])).toEqual([
      '/api/auth/register',
      '/api/auth/login',
      '/api/auth/me',
    ])
    expect(onSignedIn).toHaveBeenCalledWith({ id: 1, username: 'alice' })
  })

  it('shows the server error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ detail: 'Wrong username or password' }), { status: 401 })),
    )
    render(<AuthForm onSignedIn={vi.fn()} />)

    await userEvent.type(screen.getByLabelText('Username'), 'alice')
    await userEvent.type(screen.getByLabelText('Password'), 'nope')
    await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('Wrong username or password')
  })
})

describe('EvaluationView', () => {
  const mode = (r1: number, mrr: number, p50: number) => ({
    recall_at_1: r1,
    recall_at_5: 1,
    mrr_at_5: mrr,
    unanswerable_rejected: 0.417,
    latency_p50_ms: p50,
    latency_p95_ms: p50 * 2,
    by_type: {},
    missed: [],
  })
  const results: EvaluationResults = {
    retrieval: {
      generated_at: '2026-10-26T10:00:00+00:00',
      config: { embedding_model: 'mini', rerank_model: 'ce', chunk_size: 800, top_k: 5 },
      questions: 70,
      answerable: 58,
      unanswerable: 12,
      modes: {
        semantic: mode(0.81, 0.89, 32),
        hybrid: mode(0.828, 0.908, 30),
        hybrid_rerank: mode(0.948, 0.974, 1976),
      },
    },
    answers: null,
  }

  it('highlights the best mode and lists every mode', () => {
    render(<EvaluationView results={results} />)

    expect(screen.getByText('Best mode: Hybrid + rerank')).toBeInTheDocument()
    expect(screen.getAllByText('94.8%').length).toBeGreaterThan(0)
    expect(screen.getByText(/70 questions \(58 answerable/)).toBeInTheDocument()
    expect(screen.getByText(/Not measured yet/)).toBeInTheDocument()
  })

  it('explains how to produce results when there are none', () => {
    render(<EvaluationView results={{ retrieval: null, answers: null }} />)

    expect(screen.getByText(/No results yet/)).toBeInTheDocument()
  })
})

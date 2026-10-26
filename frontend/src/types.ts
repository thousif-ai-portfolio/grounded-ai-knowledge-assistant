// Mirrors the API schemas in app/schemas.py.

export type SearchMode = 'semantic' | 'hybrid' | 'hybrid_rerank'

export interface User {
  id: number
  username: string
}

export interface DocumentInfo {
  doc_id: string
  filename: string
  pages: number
  chunks: number
}

export interface UploadResult {
  doc_id: string
  filename: string
  chunks: number
  already_indexed: boolean
}

export interface Citation {
  number: number
  filename: string
  page: number
  text: string
  score: number
}

export interface Answer {
  answer: string
  citations: Citation[]
  grounded: boolean
  mode: SearchMode
  retrieval_ms: number
  generation_ms: number
  input_tokens: number
  output_tokens: number
}

export interface HistoryItem {
  id: number
  question: string
  answer: string
  citations: Citation[]
  grounded: boolean
  mode: SearchMode
  created_at: string
}

export interface ModeResult {
  recall_at_1: number
  recall_at_5: number
  mrr_at_5: number
  unanswerable_rejected: number
  latency_p50_ms: number
  latency_p95_ms: number
  by_type: Record<string, { questions: number; recall_at_5: number; mrr_at_5: number }>
  missed: string[]
}

export interface RetrievalReport {
  generated_at: string
  config: Record<string, string | number>
  questions: number
  answerable: number
  unanswerable: number
  modes: Record<SearchMode, ModeResult>
}

export interface AnswersReport {
  generated_at: string
  mode: SearchMode
  config: Record<string, string | number>
  metrics: {
    answered_rate: number
    correctness: number
    faithfulness: number
    citation_accuracy: number
    refusal_accuracy: number
    judge_parse_failures: number
    latency_p50_ms: number
    latency_p95_ms: number
    avg_input_tokens: number
    avg_output_tokens: number
    cost_per_question_usd: number
  }
}

export interface EvaluationResults {
  retrieval: RetrievalReport | null
  answers: AnswersReport | null
}

export const MODE_LABELS: Record<SearchMode, string> = {
  semantic: 'Semantic',
  hybrid: 'Hybrid',
  hybrid_rerank: 'Hybrid + rerank',
}

import { type ChangeEvent, useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import type { DocumentInfo } from '../types'
import { secondaryButtonClass } from '../styles'
import { Card, ErrorNote } from './ui'

export function DocumentsPage({ onError }: { onError: (err: unknown) => string }) {
  const [docs, setDocs] = useState<DocumentInfo[] | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(() => {
    api
      .listDocuments()
      .then(setDocs)
      .catch((err) => setError(onError(err)))
  }, [onError])

  useEffect(load, [load])

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setError('')
    setNotice('')
    setBusy(true)
    try {
      const result = await api.uploadDocument(file)
      setNotice(
        result.already_indexed
          ? `${file.name} was already in the knowledge base.`
          : `${file.name} added (${result.chunks} chunks).`,
      )
      load()
    } catch (err) {
      setError(onError(err))
    } finally {
      setBusy(false)
    }
  }

  async function remove(doc: DocumentInfo) {
    if (!window.confirm(`Remove ${doc.filename} from the knowledge base?`)) return
    try {
      await api.deleteDocument(doc.doc_id)
      load()
    } catch (err) {
      setError(onError(err))
    }
  }

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Knowledge base</h2>
          <p className="text-sm text-slate-500">PDF, Markdown or text files, up to 20 MB. Shared by all users.</p>
        </div>
        <label className={`${secondaryButtonClass} cursor-pointer ${busy ? 'opacity-50' : ''}`}>
          {busy ? 'Uploading…' : 'Upload a file'}
          <input
            type="file"
            accept=".pdf,.md,.txt"
            className="sr-only"
            onChange={upload}
            disabled={busy}
          />
        </label>
      </div>

      <div className="mt-4 space-y-2">
        {error && <ErrorNote message={error} />}
        {notice && <p className="text-sm text-emerald-700 dark:text-emerald-400">{notice}</p>}
      </div>

      {docs === null ? (
        <p className="mt-4 text-sm text-slate-500">Loading…</p>
      ) : docs.length === 0 ? (
        <p className="mt-4 text-sm text-slate-500">
          No documents yet. Upload one, or run <code>python -m app.cli ingest data/sample_docs</code>.
        </p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500">
              <tr>
                <th className="py-2 pr-4">File</th>
                <th className="py-2 pr-4">Pages</th>
                <th className="py-2 pr-4">Chunks</th>
                <th className="py-2" />
              </tr>
            </thead>
            <tbody>
              {docs.map((doc) => (
                <tr key={doc.doc_id} className="border-t border-slate-200 dark:border-slate-800">
                  <td className="py-2 pr-4 font-medium">{doc.filename}</td>
                  <td className="py-2 pr-4">{doc.pages}</td>
                  <td className="py-2 pr-4">{doc.chunks}</td>
                  <td className="py-2 text-right">
                    <button
                      className="text-red-600 hover:underline dark:text-red-400"
                      onClick={() => remove(doc)}
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

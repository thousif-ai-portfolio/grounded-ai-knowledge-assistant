import { describe, expect, it, vi } from 'vitest'
import { ApiError, api, getToken, setToken } from '../api'

function mockFetch(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(status === 204 ? null : JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  )
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('api', () => {
  it('logs in with form fields and stores the token', async () => {
    const fetchMock = mockFetch(200, { access_token: 'abc', token_type: 'bearer' })

    await api.login('alice', 'pw-123456')

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/auth/login')
    expect(init.body).toBeInstanceOf(URLSearchParams)
    expect(String(init.body)).toBe('username=alice&password=pw-123456')
    expect(getToken()).toBe('abc')
  })

  it('sends the stored token on later requests', async () => {
    setToken('xyz')
    const fetchMock = mockFetch(200, [])

    await api.listDocuments()

    const headers: Headers = fetchMock.mock.calls[0][1].headers
    expect(headers.get('Authorization')).toBe('Bearer xyz')
  })

  it('turns error responses into ApiError with the server message', async () => {
    mockFetch(401, { detail: 'Wrong username or password' })

    const error = await api.login('alice', 'bad').catch((e) => e)

    expect(error).toBeInstanceOf(ApiError)
    expect(error.status).toBe(401)
    expect(error.message).toBe('Wrong username or password')
  })

  it('reads the first validation error message', async () => {
    mockFetch(422, { detail: [{ msg: 'String should have at least 8 characters' }] })

    await expect(api.register('bob', 'short')).rejects.toThrow('at least 8 characters')
  })

  it('handles empty 204 responses', async () => {
    mockFetch(204, null)

    await expect(api.clearHistory()).resolves.toBeUndefined()
  })
})

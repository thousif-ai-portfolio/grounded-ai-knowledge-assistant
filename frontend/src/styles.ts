export const inputClass =
  'w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm outline-none ' +
  'focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 ' +
  'dark:border-slate-700 dark:bg-slate-900'

export const primaryButtonClass =
  'rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 ' +
  'disabled:cursor-not-allowed disabled:opacity-50'

export const secondaryButtonClass =
  'rounded-lg border border-slate-300 px-3 py-1.5 text-sm hover:bg-slate-100 ' +
  'dark:border-slate-700 dark:hover:bg-slate-800 disabled:opacity-50'

export const percent = (value: number) => `${(value * 100).toFixed(1)}%`

import React from 'react';

export const cn = (...c) => c.filter(Boolean).join(' ');

export const Card = ({ className = '', children }) => (
  <div className={cn('rounded-3xl shadow-xl border bg-white p-6 transition-all duration-300 hover:shadow-2xl dark:bg-slate-900 dark:border-slate-800', className)}>{children}</div>
);

export const Button = ({ className = '', variant = 'default', size = 'default', disabled, children, ...props }) => {
  const base = 'relative overflow-hidden rounded-2xl transition-all duration-200 flex items-center justify-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-60 disabled:cursor-not-allowed focus:ring-brand dark:focus:ring-offset-slate-900';
  const sizes = { default: 'px-5 py-2.5 text-sm font-semibold', icon: 'p-2.5', lg: 'px-8 py-4 text-base font-semibold' };
  const variants = {
    default: 'bg-brand text-white hover:bg-brand/90',
    ghost: 'bg-transparent border border-slate-300 text-slate-700 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800/80',
    outline: 'bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 dark:bg-slate-900 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800',
    subtle: 'bg-slate-100 text-slate-800 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-100 dark:hover:bg-slate-700',
    danger: 'bg-red-100 text-red-700 hover:bg-red-200 dark:bg-red-900/40 dark:text-red-300',
  };
  return (
    <button disabled={disabled} className={cn(base, sizes[size], variants[variant], className)} {...props}>{children}</button>
  );
};

export const Badge = ({ children, className = '' }) => (
  <span className={cn('inline-flex items-center rounded-full px-3 py-1 text-xs font-medium', className)}>{children}</span>
);

export const Progress = ({ value }) => (
  <div className="w-full h-2 bg-gray-200 dark:bg-slate-800 rounded-full overflow-hidden">
    <div className="h-full bg-brand transition-all" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
  </div>
);

// Segmented page navigation used in every page header.
const TABS = [['eval', 'Live Eval'], ['batch', 'Batch Score'], ['collect', 'Collect Data']];
export const PageTabs = ({ page, onNavigate, labels }) => (
  <div className="inline-flex rounded-2xl bg-slate-100 dark:bg-slate-800 p-1">
    {TABS.map(([key, fallback]) => (
      <button
        key={key}
        onClick={() => onNavigate && onNavigate(key)}
        className={cn(
          'px-3 py-1.5 rounded-xl text-sm font-semibold transition-all',
          page === key ? 'bg-white text-brand shadow dark:bg-slate-900' : 'text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
        )}
      >{labels?.[key] || fallback}</button>
    ))}
  </div>
);

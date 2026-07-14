import { useEffect, useRef, useState, useCallback } from "react";

// Live polling: calls `fn` now and every `ms` (when live). Powers the
// auto-refreshing monitoring panels.
export function usePoll<T>(fn: () => Promise<T>, ms = 3000, live = true) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const refresh = useCallback(() => {
    fnRef.current()
      .then((d) => { setData(d); setError(null); })
      .catch((e) => setError(e))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
    if (!live) return;
    const id = setInterval(refresh, ms);
    return () => clearInterval(id);
  }, [ms, live, refresh]);

  return { data, error, loading, refresh };
}

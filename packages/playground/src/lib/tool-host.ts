import { createContext } from 'react';

// Keep dropdown portals inside the embedded tool's style boundary.
export const ToolHostContext = createContext<HTMLElement | undefined>(undefined);

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import zhCN from './locales/zh-CN.json';
import en from './locales/en.json';

const storageKey = 'ascend-h3-language';
let initialLanguage = 'zh-CN';
try {
  const saved = localStorage.getItem(storageKey);
  if (saved === 'zh-CN' || saved === 'en') initialLanguage = saved;
} catch {
  // The page remains usable when browser storage is unavailable.
}

i18n.use(initReactI18next).init({
  resources: { 'zh-CN': { translation: zhCN }, en: { translation: en } },
  lng: initialLanguage,
  fallbackLng: 'zh-CN',
  supportedLngs: ['zh-CN', 'en'],
  interpolation: { escapeValue: false },
  initImmediate: false,
});

function applyLanguage(language) {
  document.documentElement.lang = language;
  document.title = i18n.t('meta.title');
  document.querySelector('meta[name="description"]')?.setAttribute('content', i18n.t('meta.description'));
  try { localStorage.setItem(storageKey, language); } catch { /* Storage is optional. */ }
}

i18n.on('languageChanged', applyLanguage);
applyLanguage(initialLanguage);

export default i18n;

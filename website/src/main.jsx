import React, { lazy, Suspense, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { motion, useReducedMotion } from 'motion/react';
import { useTranslation } from 'react-i18next';
import { ArrowDown, ArrowRight, ArrowUpRight, Check, Copy, Github } from 'lucide-react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';
import FoldText from './react-bits/FoldText/FoldText';
import FadeContent from './react-bits/FadeContent/FadeContent';
import Magnet from './react-bits/Magnet/Magnet';
import SpecularButton from './react-bits/SpecularButton/SpecularButton';
import CardNav from './react-bits/CardNav/CardNav';
import videoManifest from '../videos-manifest.json';
import './i18n';
import './style.css';

gsap.registerPlugin(ScrollTrigger);
const Galaxy = lazy(() => import('./react-bits/Galaxy/Galaxy'));
const SwarmCursor = lazy(() => import('./react-bits/SwarmCursor/SwarmCursor'));

const github = 'https://github.com/Huage001/Ascend-H3';
const showcaseSource = `${github}#watch-the-fair-comparison`;
const videoVersions = new Map(videoManifest.map(video => [video.file, video.sha256.slice(0, 12)]));
const media = [
  { key: 'full', name: 'Full + LightX2V8', bf16: 'e59cf141-3d11-44fe-911c-0ebc9c70a3f2', int8: 'b848532f-adcf-407b-99d8-23befe85ae18' },
  { key: 'vsa', name: 'VSA · FastH3 V2 · 80%', bf16: '47802ca4-31f4-46d9-a479-aef7075e3c41', int8: 'db092234-12a4-48d6-ac34-4271f97d6631' },
  { key: 'veda', name: 'Veda · 90%', bf16: 'f7120a47-795a-47b0-9ce6-7e2ef732106e', int8: 'fd7a11f1-7a7d-45ab-a9c2-2d1fb15d620a' },
  { key: 'vdn', name: 'VDN · 官方配置', bf16: '2830941f-3a21-400e-aa9c-d0bbf7b84c4c', int8: 'fa7bff37-ba41-491f-b519-c071728cf34c' },
  { key: 'spark', name: 'Spark-H3 + LightX2V8', bf16: '31f6b343-e897-4998-8969-9ff84385194c', int8: '18db86f3-4dd9-40de-89b2-64a981d709cd' },
];
function Action({ children, ...props }) {
  return <Magnet padding={36} magnetStrength={4} maxOffset={8} wrapperClassName="action-magnet"><SpecularButton radius={8} size="md" baseColor="#6b7280" tint="#fff" tintOpacity={0.06} {...props}>{children}</SpecularButton></Magnet>;
}

function ComparisonVideo({ name, precision, id, local }) {
  const { t } = useTranslation();
  const [failed, setFailed] = useState(false);
  const original = `https://github.com/user-attachments/assets/${id}`;
  const src = local ? `${import.meta.env.BASE_URL}videos/${local}?v=${videoVersions.get(local)}` : original;
  return <div className="video-item"><div className="video-meta"><span>{name}</span><strong>{precision}</strong></div><div className="video-frame">{failed ? <div className="video-unavailable"><span>{t('videos.unavailable')}</span><a href={original} target="_blank" rel="noreferrer">{t('videos.github')} <ArrowUpRight size={14} /></a></div> : <video controls playsInline preload="metadata" src={src} onError={() => setFailed(true)} aria-label={t('videos.aria', { name, precision })} />}</div></div>;
}

function MethodDiagram({ name, diagram, t }) {
  const block = (node) => <div className={`flow-block${node.kind ? ` flow-block--${node.kind}` : ''}`}><strong>{node.title}</strong>{node.detail && <span>{node.detail}</span>}</div>;
  return <figure className="method-diagram" aria-label={t('methods.path', { name })}>
    <div className="diagram-heading"><span>{t('diagram.eyebrow')}</span><strong>{name} / ASCEND</strong></div>
    <div className="flowchart"><div className="flow-terminal"><span>{t('diagram.inputLabel')}</span><strong>{diagram.input}</strong></div>
      {diagram.steps.map(step => <React.Fragment key={step.title}><div className="flow-arrow" aria-hidden="true" />{step.branches ? <div className="flow-branches" aria-label={t('diagram.branches')}>{step.branches.map(branch => <div className="flow-branch" key={branch.title}>{block(branch)}</div>)}</div> : block(step)}</React.Fragment>)}
      <div className="flow-arrow" aria-hidden="true" /><div className="flow-terminal flow-terminal--output"><span>{t('diagram.outputLabel')}</span><strong>{diagram.output}</strong></div>
    </div>
    <figcaption>{t('diagram.caption')}</figcaption>
  </figure>;
}
const methods = [
  { key: 'full', name: 'Full', number: '01', source: 'src/h3_ascend/backends/full.py', color: '#e1e8d6' },
  { key: 'vsa', name: 'VSA', number: '02', source: 'csrc/vsa_paired64/README.md', color: '#9dc9ac' },
  { key: 'veda', name: 'Veda', number: '03', source: 'src/h3_ascend/backends/veda.py', color: '#e0bb85' },
  { key: 'vdn', name: 'VDN', number: '04', source: 'src/h3_ascend/backends/vdn.py', color: '#c9b6dd' },
  { key: 'spark', name: 'Spark-H3', number: '05', source: 'src/h3_ascend/backends/spark.py', color: '#a9c8d3' },
];

const coreBenchmark = {
  124: [
    { name: 'Full', ascend: [343.31, 1.00], gpu: [111.64, 1.00] },
    { name: 'VSA', ascend: [216.24, 1.59], gpu: [93.56, 1.19] },
    { name: 'Veda', ascend: [75.50, 4.55], gpu: [25.01, 4.46] },
    { name: 'VDN', ascend: [368.89, 0.93], gpu: [142.74, 0.78] },
    { name: 'Spark-H3', ascend: [128.54, 2.67], gpu: [39.35, 2.84] },
  ],
  345: [
    { name: 'Full', ascend: [2578.60, 1.00], gpu: [838.10, 1.00] },
    { name: 'VSA', ascend: [1196.30, 2.16], gpu: [498.34, 1.68] },
    { name: 'Veda', ascend: [380.26, 6.78], gpu: [123.45, 6.79] },
    { name: 'VDN', ascend: [1093.46, 2.36], gpu: [423.12, 1.98] },
    { name: 'Spark-H3', ascend: [636.98, 4.05], gpu: [184.38, 4.55] },
  ],
};

const benchmark = {
  124: [
    { name: 'Full', single: 197.66, multi: 28.38, speed: '6.96', note: '†T' },
    { name: 'VSA', single: 146.81, multi: 22.19, speed: '6.61', note: '†T' },
    { name: 'Veda', single: 97.40, multi: 17.22, speed: '5.66', note: '†T' },
    { name: 'VDN', single: 216.91, multi: 34.91, speed: '6.21', note: '†T' },
    { name: 'Spark-H3', single: 117.11, multi: 20.78, speed: '5.64', note: '†T' },
  ],
  345: [
    { name: 'Full', single: 1211.17, multi: 159.88, speed: '7.58', note: '†T' },
    { name: 'VSA', single: 617.55, multi: 83.26, speed: '7.42', note: '†L' },
    { name: 'Veda', single: 337.95, multi: 54.98, speed: '6.15', note: '†T' },
    { name: 'VDN', single: 620.67, multi: 92.20, speed: '6.73', note: '†L' },
    { name: 'Spark-H3', single: 419.44, multi: 66.18, speed: '6.34', note: '†L' },
  ],
};

function LatencyChart({ frameCount, devices, rows, t }) {
  const isSingle = devices === 'single';
  const maximum = frameCount === 124 ? 230 : 1250;
  return <figure className="latency-chart" aria-labelledby={`latency-${frameCount}-${devices}-frames latency-${frameCount}-${devices}`}>
    <figcaption className="latency-heading"><div><span id={`latency-${frameCount}-${devices}-frames`}>{t('performance.frame', { frames: frameCount })}</span><h4 id={`latency-${frameCount}-${devices}`}>{t(isSingle ? 'chart.single' : 'chart.multi')}</h4></div><span className="latency-unit">{t('chart.unit')}</span></figcaption>
    <ul className="latency-rows">{rows.map(row => {
      const value = isSingle ? row.single : row.multi;
      const color = methods.find(item => item.name === row.name)?.color ?? '#cbd7c8';
      return <li className="latency-row" key={row.name}>
        <span className="latency-name">{row.name}</span>
        <span className="latency-track" aria-hidden="true"><span className="latency-bar" style={{ width: `${value / maximum * 100}%`, backgroundColor: color }} /></span>
        <span className="latency-value">{value.toFixed(2)}{isSingle && <small> {row.note}</small>}</span>
      </li>;
    })}</ul>
    <div className="latency-axis" aria-hidden="true"><span>0</span><span>{t('chart.axisMaximum', { value: maximum })}</span></div>
  </figure>;
}

function App() {
  const { t, i18n } = useTranslation();
  const [coreFrames, setCoreFrames] = useState(124);
  const [copied, setCopied] = useState(false);
  const reduced = useReducedMotion();
  const navItems = [
    { label: t('nav.project'), description: t('nav.projectDescription'), links: [{ label: t('nav.overview'), href: '#overview' }, { label: t('nav.methods'), href: '#methods' }] },
    { label: t('nav.results'), description: t('nav.resultsDescription'), links: [{ label: t('nav.performance'), href: '#performance' }, { label: t('nav.videos'), href: '#videos' }] },
    { label: t('nav.start'), description: t('nav.startDescription'), links: [{ label: t('nav.quickstart'), href: '#start' }, { label: 'GitHub', href: github }] },
  ];
  const command = 'h3-ascend serve --method full --model /models/MiniMax-H3 \\\n  --precision int8 --devices 1 --offload text';

  useEffect(() => {
    if (reduced) return;
    const lenis = new Lenis({ anchors: true, lerp: 0.09 });
    const update = (time) => lenis.raf(time * 1000);
    lenis.on('scroll', ScrollTrigger.update);
    gsap.ticker.add(update);
    gsap.ticker.lagSmoothing(0);
    const context = gsap.context(() => {
      gsap.utils.toArray('.reveal').forEach((el) => {
        gsap.fromTo(el, { y: 32, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power2.out', scrollTrigger: { trigger: el, start: 'top 88%', once: true } });
      });
    });
    return () => { context.revert(); gsap.ticker.remove(update); lenis.destroy(); };
  }, [reduced]);

  useEffect(() => {
    ScrollTrigger.refresh();
  }, [i18n.resolvedLanguage]);

  async function copyCommand() {
    try { await navigator.clipboard.writeText(command); setCopied(true); window.setTimeout(() => setCopied(false), 2000); } catch { setCopied(false); }
  }

  return <>
    <div className="galaxy-background" aria-hidden="true"><Suspense fallback={null}><Galaxy density={0.65} starSpeed={0.18} speed={0.35} glowIntensity={0.18} saturation={0.2} hueShift={165} layers={3} starSize={0.7} flareIntensity={0.35} clickWarp={false} disableAnimation={!!reduced} /></Suspense></div>
    <header className="site-header"><CardNav logo={<span className="nav-brand"><img src={`${import.meta.env.BASE_URL}go-there-lab-logo.png`} alt="" /><span>Go There Lab<small>ASCEND-H3</small></span></span>} logoAlt={t('nav.home')} logoHref="#top" items={navItems} maxWidth={1320} intro={false} ariaLabel={t('nav.aria')} menuText={t('nav.menu')} closeText={t('nav.close')} /><div className="language-switch" role="group" aria-label={t('language.label')}><button type="button" lang="zh-CN" aria-pressed={i18n.resolvedLanguage === 'zh-CN'} onClick={() => i18n.changeLanguage('zh-CN')}>中文</button><button type="button" lang="en" aria-pressed={i18n.resolvedLanguage === 'en'} onClick={() => i18n.changeLanguage('en')}>EN</button></div></header>

    <main id="top">
      <section className="hero">
        <div className="swarm-layer" aria-hidden="true">{!reduced && <Suspense fallback={null}><SwarmCursor count={420} size={2} spread={90} color="#d9e9e3" accentColor="#acc7bd" idle="roam" clickEffect="none" /></Suspense>}</div>
        <div className="hero-content wrap">
          <motion.div initial={reduced ? false : { opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <img className="hero-lab-mark" src={`${import.meta.env.BASE_URL}go-there-lab-logo.png`} alt="Go There Lab" />
            <div className="eyebrow light"><span className="status-dot" /> {t('labels.hero')}</div>
            <h1><FoldText text="Ascend-H3" fontSize="inherit" splitBy="char" stagger={0.045} duration={0.8} /></h1>
            <p className="hero-cn">{t('hero.headline')}</p>
            <p className="hero-desc">{t('hero.description1')}<br />{t('hero.description2')}</p>
            <div className="hero-actions"><Action as="a" href={github} target="_blank" rel="noreferrer"><Github size={17} /> {t('hero.explore')} <ArrowUpRight size={16} /></Action><Action as="a" href="#videos">{t('hero.watch')} <ArrowDown size={16} /></Action></div>
          </motion.div>
          <div className="hero-bottom"><span>MINIMAX H3 / ASCEND 910B3</span><span>{t('hero.scroll')}</span></div>
        </div>
      </section>

      <section id="overview" className="intro section-pad">
        <div className="wrap intro-grid reveal"><div className="section-kicker"><span>{t('labels.project')}</span><span className="kicker-line" /></div><div><h2>{t('overview.heading1')}<br /><span>{t('overview.heading2')}</span></h2><p className="intro-copy">{t('overview.intro')}</p><p className="body-copy">{t('overview.body1')}</p><p className="body-copy">{t('overview.body2')}</p><a className="inline-link" href={`${github}#readme`} target="_blank" rel="noreferrer">{t('overview.docs')} <ArrowUpRight size={16} /></a></div></div>
        <div className="wrap facts reveal"><div><strong>05</strong><span>{t('overview.methods')}</span></div><div><strong>768<span>p</span></strong><span>{t('overview.resolution')}</span></div><div><strong>1/2/4/8</strong><span>{t('overview.devices')}</span></div><div><strong>TP / SP</strong><span>{t('overview.parallel')}</span></div></div>
        <div className="wrap integration-line reveal"><div><span>{t('labels.execution')}</span><h3>{t('overview.executionTitle')}</h3><p>{t('overview.execution')}</p></div><div><span>{t('labels.precision')}</span><h3>{t('overview.precisionTitle')}</h3><p>{t('overview.precision')}</p></div><div><span>{t('labels.generation')}</span><h3>{t('overview.generationTitle')}</h3><p>{t('overview.generation')}</p></div></div>
      </section>

      <section id="methods" className="methods section-pad">
        <div className="wrap"><div className="section-top reveal"><div><div className="section-kicker"><span>{t('labels.methods')}</span><span className="kicker-line" /></div><h2>{t('methods.heading1')}<br />{t('methods.heading2')}</h2></div><p>{t('methods.intro')}</p></div>
          <div className="methods-grid reveal">{methods.map(item => {
            const copy = t(`methods.items.${item.key}`, { returnObjects: true });
            const diagram = t(`diagram.${item.key}`, { returnObjects: true });
            return <article className="method-item" key={item.key} style={{ '--method-color': item.color }} aria-labelledby={`method-${item.key}`}>
              <header className="method-item-heading"><span className="detail-label">{copy.label}</span><h3 id={`method-${item.key}`}>{item.name}</h3></header>
              <p className="method-description">{copy.description}</p>
              <MethodDiagram name={item.name} diagram={diagram} t={t} />
              <div className="method-foot"><p>{copy.note}</p><a className="inline-link" href={`${github}/blob/main/${item.source}`} target="_blank" rel="noreferrer">{t('methods.source')} <ArrowUpRight size={15} /></a></div>
            </article>;
          })}</div>
        </div>
      </section>

      <section id="performance" className="performance section-pad">
        <div className="wrap"><div className="section-top reveal"><div><div className="section-kicker"><span>{t('labels.performance')}</span><span className="kicker-line" /></div><h2>{t('performance.heading1')}<br /><span>{t('performance.heading2')}</span></h2></div><p>{t('performance.intro')}</p></div>
          <div className="core-comparison reveal"><div className="benchmark-head"><div><span className="micro">{t('labels.core')}</span><h3>{t('performance.coreTitle')}</h3></div><div className="segmented" aria-label={t('performance.coreFrames')}>{[124, 345].map(n => <SpecularButton key={n} radius={6} size="sm" className={coreFrames === n ? 'selected' : ''} aria-pressed={coreFrames === n} onClick={() => setCoreFrames(n)}>{t('performance.frame', { frames: n })}</SpecularButton>)}</div></div><p className="core-intro">{t('performance.coreIntro')}<strong>{t('performance.coreWarning')}</strong></p><div className="core-table-wrap"><table className="core-table"><thead><tr><th scope="col">{t('performance.method')}</th><th scope="col">{t('performance.ascend')} <span>{t('performance.latencySpeed')}</span></th><th scope="col">RTX PRO 6000 <span>{t('performance.latencySpeed')}</span></th></tr></thead><tbody>{coreBenchmark[coreFrames].map(row => <tr key={row.name}><th scope="row">{row.name}</th><td><span>{row.ascend[0].toFixed(2)} ms</span><strong>{row.ascend[1].toFixed(2)}×</strong></td><td><span>{row.gpu[0].toFixed(2)} ms</span><strong>{row.gpu[1].toFixed(2)}×</strong></td></tr>)}</tbody></table></div><p className="benchmark-note">{t('performance.coreNote')}</p><a className="inline-link" href={`${github}/blob/main/benchmarks/bf16-three-layer.json`} target="_blank" rel="noreferrer">{t('performance.coreSource')} <ArrowUpRight size={16} /></a></div>
          <div className="benchmark reveal"><div className="benchmark-head"><div><span className="micro">{t('labels.e2e')}</span><h3>{t('performance.e2eTitle')}</h3></div></div>
            <p className="latency-explainer">{t('chart.explainer')}</p>
            <div className="latency-grid">{[124, 345].flatMap(frameCount => ['single', 'multi'].map(devices => <LatencyChart key={`${frameCount}-${devices}`} frameCount={frameCount} devices={devices} rows={benchmark[frameCount]} t={t} />))}</div>
            <p className="benchmark-note">{t('performance.e2eNote')} {t('chart.offloadComparison')}</p><a className="inline-link" href={`${github}#performance`} target="_blank" rel="noreferrer">{t('performance.e2eSource')} <ArrowUpRight size={16} /></a>
          </div>
        </div>
      </section>

      <section id="videos" className="evidence section-pad"><div className="wrap"><div className="section-top reveal"><div><div className="section-kicker"><span>{t('labels.videos')}</span><span className="kicker-line" /></div><h2>{t('videos.heading1')}<br />{t('videos.heading2')}</h2></div><p>{t('videos.intro')}</p></div><div className="video-heading"><span>{t('labels.videoMethods')}</span><span>BF16 <span className="heading-slash">/</span> INT8 W8A8</span></div><div className="video-list">{media.map((item, i) => { const name = item.key === 'vdn' ? t('videos.vdnName') : item.name; return <FadeContent key={item.key} duration={0.8} blur={true} threshold={0.08}><div className="comparison-row"><div className="comparison-title"><span>{String(i + 1).padStart(2, '0')}</span><h3>{name}</h3></div><div className="video-pair"><ComparisonVideo name={name} precision="BF16" id={item.bf16} local={item.key + '-bf16.mp4'} /><ComparisonVideo name={name} precision="INT8 · W8A8" id={item.int8} local={item.key + '-int8.mp4'} /></div></div></FadeContent>; })}</div><p className="video-note">{t('videos.note')}</p><a className="inline-link" href={showcaseSource} target="_blank" rel="noreferrer">{t('videos.source')} <ArrowUpRight size={16} /></a></div></section>

      <section id="start" className="start section-pad"><div className="wrap start-grid reveal"><div><div className="section-kicker"><span>{t('labels.start')}</span><span className="kicker-line" /></div><h2>{t('start.heading1')}<br />{t('start.heading2')}</h2><p>{t('start.intro')}</p><a className="inline-link light-link" href={`${github}/blob/main/docs/environment.md`} target="_blank" rel="noreferrer">{t('start.environment')} <ArrowUpRight size={16} /></a></div><div className="terminal"><div className="terminal-head"><span><i /> {t('labels.terminal')}</span><SpecularButton radius={6} size="sm" onClick={copyCommand} aria-label={copied ? t('start.copied') : t('start.copy')} title={copied ? t('start.copiedTitle') : t('start.copy')}>{copied ? <Check size={17} /> : <Copy size={17} />}</SpecularButton></div><div className="terminal-body"><span className="terminal-comment">{t('start.comment')}</span><pre><code>{command}</code></pre><span className="terminal-hint">{t('start.hint')}</span></div><a href={`${github}#serve`} target="_blank" rel="noreferrer">{t('start.guide')} <ArrowRight size={16} /></a></div></div></section>
    </main>
    <footer><div className="wrap footer-inner"><div><strong>Ascend-H3<span>.</span></strong><p>{t('labels.footer')}</p></div><div className="footer-links"><a href={github} target="_blank" rel="noreferrer">GitHub <ArrowUpRight size={14} /></a><a href={`${github}/blob/main/LICENSE`} target="_blank" rel="noreferrer">{t('footer.license')} <ArrowUpRight size={14} /></a><a href={`${import.meta.env.BASE_URL}licenses/react-bits.txt`} target="_blank" rel="noreferrer">React Bits <ArrowUpRight size={14} /></a><a href="mailto:liusonghua@sjtu.edu.cn">{t('footer.contact')} <ArrowUpRight size={14} /></a></div><small>{t('footer.notice')}</small></div></footer>
  </>;
}

const root = createRoot(document.getElementById('root'));
root.render(<App />);
if (import.meta.hot) import.meta.hot.dispose(() => root.unmount());

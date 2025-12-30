/**
 * HTML to RAG-Optimized Markdown Converter
 * Based on: https://github.com/Gafoor2005/HTML-to-RAG-Optimized-Markdown
 * 
 * Converts HTML to Markdown with semantic hierarchy and interactive element
 * metadata for RAG indexing. Preserves identifiers (class, id, etc.) for
 * interactive elements.
 * 
 * Usage: Reads HTML from stdin, outputs Markdown to stdout
 *   echo "<html>...</html>" | node html_to_rag_markdown.js
 */

const { JSDOM } = require('jsdom');

// Read HTML from stdin
let html = '';
process.stdin.setEncoding('utf8');

process.stdin.on('readable', () => {
    let chunk;
    while ((chunk = process.stdin.read()) !== null) {
        html += chunk;
    }
});

process.stdin.on('end', () => {
    const output = convertHtmlToRagMarkdown(html);
    process.stdout.write(output);
});

function convertHtmlToRagMarkdown(html) {
    const dom = new JSDOM(html);
    const document = dom.window.document;

    // Extract page title
    let pageTitle = '';
    const titleEl = document.querySelector('title');
    if (titleEl) {
        pageTitle = (titleEl.textContent || '').trim();
    }
    // Fallback to og:title meta tag
    if (!pageTitle) {
        const ogTitle = document.querySelector('meta[property="og:title"]');
        if (ogTitle) {
            pageTitle = (ogTitle.getAttribute('content') || '').trim();
        }
    }
    // Final fallback
    if (!pageTitle) {
        pageTitle = 'Page Content';
    }

    function isHeadingTag(el) {
        if (!el || !el.tagName) return false;
        const tag = el.tagName.toLowerCase();
        return ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(tag);
    }

    function isSectionElement(el) {
        if (!el || !el.tagName) return false;
        const tag = el.tagName.toLowerCase();
        return ['header', 'main', 'footer', 'nav', 'aside', 'section', 'article'].includes(tag);
    }

    function getSectionLabel(el) {
        const tag = el.tagName.toLowerCase();
        let label = tag.charAt(0).toUpperCase() + tag.slice(1);
        const attrs = [];

        // Add aria-label if present
        const ariaLabel = (el.getAttribute('aria-label') || '').trim();
        if (ariaLabel) {
            attrs.push(`label="${ariaLabel}"`);
        }

        // Add aria-labelledby if present
        const ariaLabelledby = (el.getAttribute('aria-labelledby') || '').trim();
        if (ariaLabelledby) {
            attrs.push(`labelledby="${ariaLabelledby}"`);
        }

        // Add role if present
        const role = (el.getAttribute('role') || '').trim();
        if (role) {
            attrs.push(`role="${role}"`);
        }

        // Add id if present
        const id = (el.getAttribute('id') || '').trim();
        if (id) {
            attrs.push(`id="${id}"`);
        }

        // Add class if present (filter out utility classes)
        const cls = (el.getAttribute('class') || '').trim();
        if (cls) {
            const classes = cls.split(/\s+/).filter(c => {
                // Filter out common utility classes
                return !c.match(/^(sc-|g-|l-|m-|theme-|left|right|fixed|active|visible|hidden)/);
            });
            if (classes.length > 0 && classes.length <= 3) {
                attrs.push(`class="${classes.join(' ')}"`);
            }
        }

        // Add other ARIA attributes that provide semantic meaning
        const semanticAria = [
            'aria-describedby', 'aria-owns', 'aria-controls',
            'aria-live', 'aria-atomic', 'aria-relevant',
            'aria-expanded', 'aria-haspopup', 'aria-current'
        ];
        for (const attrName of semanticAria) {
            const val = (el.getAttribute(attrName) || '').trim();
            if (val) {
                attrs.push(`${attrName.replace('aria-', '')}="${val}"`);
            }
        }

        // Add important data-* attributes
        if (el.attributes) {
            for (const attr of Array.from(el.attributes)) {
                if (attr.name.startsWith('data-') && attr.value) {
                    // Include only shorter, more meaningful data attributes
                    if (attr.value.length < 50) {
                        attrs.push(`${attr.name}="${attr.value}"`);
                    }
                }
            }
        }

        if (attrs.length > 0) {
            label += ` [${attrs.join(' ')}]`;
        }

        return label;
    }

    function isSkippable(node) {
        const parent = node.parentElement;
        if (!parent) return false;
        const tag = parent.tagName ? parent.tagName.toLowerCase() : '';
        if (tag === 'script' || tag === 'style' || tag === 'iframe') return true;
        // Check if inside an iframe
        let ancestor = parent;
        while (ancestor) {
            if (ancestor.tagName && ancestor.tagName.toLowerCase() === 'iframe') return true;
            ancestor = ancestor.parentElement;
        }
        return false;
    }

    function isInteractive(el) {
        if (!el || !el.tagName) return false;
        const tag = el.tagName.toLowerCase();
        if (['a', 'button', 'input', 'select', 'textarea', 'label', 'summary'].includes(tag)) return true;
        const role = (el.getAttribute('role') || '').toLowerCase();
        if (role && ['button', 'link', 'checkbox', 'switch', 'menuitem', 'tab', 'combobox', 'progressbar', 'slider'].includes(role)) return true;
        return false;
    }

    function normalizeText(t) {
        return t.replace(/\s+/g, ' ').trim();
    }

    function getLabelFromIds(root, ids) {
        const parts = [];
        for (const id of ids) {
            const ref = root.querySelector('#' + id) || document.getElementById(id);
            if (ref) {
                const t = normalizeText((ref.textContent || '').trim());
                if (t) parts.push(t);
            }
        }
        return parts.join(' ');
    }

    function accessibleName(el) {
        // aria-label
        const aria = (el.getAttribute('aria-label') || '').trim();
        if (aria) return aria;
        // aria-labelledby
        const labelledBy = (el.getAttribute('aria-labelledby') || '').trim();
        if (labelledBy) {
            const ids = labelledBy.split(/\s+/).filter(Boolean);
            const viaIds = getLabelFromIds(document, ids);
            if (viaIds) return viaIds;
        }
        // title attribute
        const title = (el.getAttribute('title') || '').trim();
        if (title) return title;
        // svg content
        const svg = el.tagName.toLowerCase() === 'svg' ? el : el.querySelector('svg');
        if (svg) {
            const svgAria = (svg.getAttribute('aria-label') || '').trim();
            if (svgAria) return svgAria;
            const svgLabelledBy = (svg.getAttribute('aria-labelledby') || '').trim();
            if (svgLabelledBy) {
                const ids = svgLabelledBy.split(/\s+/).filter(Boolean);
                const viaIds = getLabelFromIds(svg, ids);
                if (viaIds) return viaIds;
            }
            const svgTitleEl = svg.querySelector('title');
            if (svgTitleEl) {
                const t = normalizeText((svgTitleEl.textContent || '').trim());
                if (t) return t;
            }
            const svgDescEl = svg.querySelector('desc');
            if (svgDescEl) {
                const d = normalizeText((svgDescEl.textContent || '').trim());
                if (d) return d;
            }
        }
        // img alt inside
        const img = el.querySelector && el.querySelector('img[alt]');
        if (img) {
            const alt = (img.getAttribute('alt') || '').trim();
            if (alt) return alt;
        }
        // visible text
        const txt = normalizeText((el.textContent || '').trim());
        if (txt) return txt;
        return '';
    }

    function collectAttrs(el) {
        const attrs = [];
        const stateNames = [
            'aria-expanded',
            'aria-pressed',
            'aria-selected',
            'aria-checked',
            'aria-disabled',
            'aria-current',
            'aria-busy',
        ];
        const states = [];
        for (const name of stateNames) {
            const v = el.getAttribute(name);
            if (v !== null) states.push(`${name.replace('aria-', '')}=${v}`);
        }
        const statePrefix = states.length ? `[STATE ${states.join(' ')}] ` : '';

        const tag = el.tagName.toLowerCase();
        const cls = (el.getAttribute('class') || '').trim();
        const id = (el.getAttribute('id') || '').trim();
        attrs.push(`element=${tag}`);
        if (cls) attrs.push(`class="${cls}"`);
        if (id) attrs.push(`id="${id}"`);

        if (tag === 'a') {
            const href = el.getAttribute('href') || '';
            const title = el.getAttribute('title') || '';
            if (href) attrs.push(`href="${href}"`);
            if (title) attrs.push(`title="${title}"`);
        }
        if (tag === 'button' || tag === 'input') {
            const type = el.getAttribute('type') || '';
            const label = el.getAttribute('title') || el.getAttribute('aria-label') || '';
            if (type) attrs.push(`type="${type}"`);
            if (label) attrs.push(`label="${label}"`);
        }
        const role = el.getAttribute('role') || '';
        if (role) attrs.push(`role="${role}"`);

        if (el.attributes) {
            for (const attr of Array.from(el.attributes)) {
                const name = attr.name;
                if (name.startsWith('aria-') || name.startsWith('data-')) {
                    if (!stateNames.includes(name)) attrs.push(`${name}="${attr.value}"`);
                }
            }
        }

        return { statePrefix, attrs };
    }

    function closestInteractive(node) {
        let el = node.parentElement;
        while (el) {
            if (isInteractive(el)) return el;
            el = el.parentElement;
        }
        return null;
    }

    const walker = document.createTreeWalker(
        document.body || document,
        dom.window.NodeFilter.SHOW_ALL
    );
    const lines = [];
    const processedHeadings = new Set();
    const processedSections = new Set();
    const sectionStack = [];

    while (walker.nextNode()) {
        const node = walker.currentNode;
        if (node.nodeType === dom.window.Node.ELEMENT_NODE) {
            const el = node;
            // Skip iframes entirely
            if (el.tagName && el.tagName.toLowerCase() === 'iframe') continue;

            // Handle section elements (header, main, footer, etc.)
            if (isSectionElement(el) && !processedSections.has(el)) {
                processedSections.add(el);
                const label = getSectionLabel(el);
                const depth = sectionStack.length;
                lines.push({ kind: 'section', label, depth });
                sectionStack.push(el);
                continue;
            }

            // Handle heading tags first - this takes priority
            if (isHeadingTag(el)) {
                if (processedHeadings.has(el)) continue; // Skip if already processed
                processedHeadings.add(el);

                const level = parseInt(el.tagName.toLowerCase()[1]);
                const depth = sectionStack.length;

                // Extract all content from the heading - both text and interactive elements
                const headingWalker = document.createTreeWalker(el, dom.window.NodeFilter.SHOW_ALL);
                const headingContent = [];

                while (headingWalker.nextNode()) {
                    const hNode = headingWalker.currentNode;

                    if (hNode.nodeType === dom.window.Node.ELEMENT_NODE) {
                        const hEl = hNode;
                        if (isInteractive(hEl)) {
                            const name = accessibleName(hEl);
                            if (name) {
                                const { statePrefix, attrs } = collectAttrs(hEl);
                                headingContent.push({ kind: 'heading-interactive', text: name, statePrefix, attrs, level, depth });
                            }
                        }
                    } else if (hNode.nodeType === dom.window.Node.TEXT_NODE) {
                        const text = normalizeText(hNode.nodeValue || '');
                        if (!text) continue;
                        // Check if this text is inside an interactive element
                        let insideInteractive = false;
                        let ancestor = hNode.parentElement;
                        while (ancestor && ancestor !== el) {
                            if (isInteractive(ancestor)) {
                                insideInteractive = true;
                                break;
                            }
                            ancestor = ancestor.parentElement;
                        }

                        if (!insideInteractive) {
                            headingContent.push({ kind: 'heading', text, level, depth });
                        }
                    }
                }

                // Add all heading content to lines
                lines.push(...headingContent);
                continue;
            }

            // Prioritize interactive nodes (but not if inside a heading)
            if (isInteractive(el)) {
                // Check if inside a heading - if so, skip (heading will handle it)
                let insideHeading = false;
                let ancestor = el.parentElement;
                while (ancestor) {
                    if (isHeadingTag(ancestor)) {
                        insideHeading = true;
                        break;
                    }
                    ancestor = ancestor.parentElement;
                }
                if (insideHeading) continue;

                const name = accessibleName(el);
                if (!name) continue;
                const { statePrefix, attrs } = collectAttrs(el);
                const depth = sectionStack.length;
                lines.push({ kind: 'interactive', text: name, statePrefix, attrs, depth });
                continue;
            }
        } else if (node.nodeType === dom.window.Node.TEXT_NODE) {
            if (isSkippable(node)) continue;
            const text = normalizeText(node.nodeValue || '');
            if (!text) continue;

            // Check if inside an interactive element - if so, skip
            const insideInteractive = closestInteractive(node);
            if (insideInteractive) continue;

            // Check if inside a heading - if so, skip (heading will handle it)
            const parent = node.parentElement;
            let insideHeading = false;
            let ancestor = parent;
            while (ancestor) {
                if (isHeadingTag(ancestor)) {
                    insideHeading = true;
                    break;
                }
                ancestor = ancestor.parentElement;
            }
            if (insideHeading) continue;

            const depth = sectionStack.length;
            lines.push({ kind: 'text', text, depth });
        }

        // Check if we're leaving a section element
        while (sectionStack.length > 0) {
            const currentSection = sectionStack[sectionStack.length - 1];
            if (!currentSection.contains(walker.currentNode) && currentSection !== walker.currentNode) {
                sectionStack.pop();
            } else {
                break;
            }
        }
    }

    let output = `# ${pageTitle}\n\n`;
    let prevKind = null;
    for (const item of lines) {
        // Add spacing when transitioning between kinds (except for section markers)
        if (prevKind !== null && prevKind !== item.kind && item.kind !== 'section') {
            output += '\n';
        }

        const indent = '    '.repeat(item.depth || 0);

        if (item.kind === 'section') {
            // Section markers use list marker + heading syntax with # count based on depth
            const hashes = '#'.repeat((item.depth || 0) + 3);
            output += `${indent}- ${hashes} ${item.label}\n`;
        } else if (item.kind === 'heading') {
            const hashes = '#'.repeat(item.level + 1); // offset by 1 since doc title is #
            output += `${indent}${hashes} ${item.text}\n\n`;
        } else if (item.kind === 'heading-interactive') {
            const hashes = '#'.repeat(item.level + 1);
            const attrs = `[${item.attrs.join(' ')}]`;
            output += `${indent}${hashes} **${item.text}** — ${attrs}\n\n`;
        } else if (item.kind === 'text') {
            output += `${indent}- ${item.text}\n`;
        } else {
            const attrs = `[${item.attrs.join(' ')}]`;
            output += `${indent}- ${item.statePrefix}**${item.text}** — ${attrs}\n`;
        }
        prevKind = item.kind;
    }

    return output;
}

module.exports = { convertHtmlToRagMarkdown };

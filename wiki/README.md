# pymodal wiki source

Markdown sources for the GitHub Wiki at <https://github.com/grcarmenaty/pymodal/wiki>.

GitHub renders one wiki page per markdown file at the root of the wiki repository (`grcarmenaty/pymodal.wiki.git`). Page titles are derived from the filename (with `-` rendered as a space). Special files:

- `Home.md` — the wiki landing page.
- `_Sidebar.md` — navigation sidebar shown on every page.
- `_Footer.md` — footer shown on every page.

## Files

| File | Page title |
|---|---|
| `Home.md` | Home |
| `Installation.md` | Installation |
| `Core-Concepts.md` | Core Concepts |
| `Quickstart.md` | Quickstart |
| `Collections.md` | Collections |
| `Timeseries.md` | Timeseries |
| `FRF.md` | FRF |
| `Indicators.md` | Indicators |
| `HDF5-Dataset.md` | HDF5 Dataset |
| `MCP-Server.md` | MCP Server |
| `API-Reference.md` | API Reference |
| `FAQ.md` | FAQ |
| `_Sidebar.md` | (sidebar, all pages) |
| `_Footer.md` | (footer, all pages) |

## Publishing to GitHub Wiki

Once the wiki has been initialised on github.com (visit `Settings → Features → Wikis` and create the first page if needed), clone the wiki repo and copy these files in:

```bash
git clone https://github.com/grcarmenaty/pymodal.wiki.git
cp wiki/*.md pymodal.wiki/
cd pymodal.wiki
git add .
git commit -m "Initial wiki content"
git push
```

The wiki repository is independent of the main repo, so updates to these files in `master` do not flow automatically; rerun the copy/commit/push when content changes.

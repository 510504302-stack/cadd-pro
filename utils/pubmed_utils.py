"""
PubMed/PMC literature search utility functions.
"""
import re
from typing import List, Dict, Optional, Tuple
from Bio import Entrez
import streamlit as st


def set_entrez_email(email: str = "user@example.com"):
    """Set Entrez email (required by NCBI)."""
    Entrez.email = email


def search_pmc(keyword: str, max_results: int = 10) -> List[str]:
    """Search PubMed Central for a keyword and return article IDs."""
    try:
        handle = Entrez.esearch(db="pmc", term=keyword, retmode="xml", retmax=max_results)
        record = Entrez.read(handle, validate=False)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        st.error(f"PMC 搜索失败: {e}")
        return []


def search_pubmed(keyword: str, max_results: int = 10) -> List[str]:
    """Search PubMed for a keyword and return article IDs."""
    try:
        handle = Entrez.esearch(db="pubmed", term=keyword, retmode="xml", retmax=max_results)
        record = Entrez.read(handle, validate=False)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        st.error(f"PubMed 搜索失败: {e}")
        return []


def fetch_article_details(pmcid: str) -> Optional[Dict]:
    """Fetch full article details from PMC by ID."""
    try:
        handle = Entrez.efetch(db="pmc", id=pmcid, retmode="text")
        record = Entrez.read(handle, validate=False)
        handle.close()
        return record
    except Exception as e:
        st.warning(f"获取文献 {pmcid} 失败: {e}")
        return None


def fetch_pubmed_summary(pubmed_ids: List[str]) -> List[Dict]:
    """Fetch summaries for PubMed articles."""
    try:
        handle = Entrez.esummary(db="pubmed", id=",".join(pubmed_ids), retmode="xml")
        records = Entrez.read(handle, validate=False)
        handle.close()
        summaries = []
        for record in records:
            summaries.append({
                'id': str(record.get('Id', '')),
                'title': str(record.get('Title', '')),
                'source': str(record.get('Source', '')),
                'pubdate': str(record.get('PubDate', '')),
                'authors': ', '.join(record.get('AuthorList', [])) if record.get('AuthorList') else 'N/A',
            })
        return summaries
    except Exception as e:
        st.warning(f"获取PubMed摘要失败: {e}")
        return []


def extract_article_text(article_details) -> Tuple[str, str, str]:
    """Extract title, abstract, and full text from PMC article details."""
    title = ""
    abstract = ""
    full_text = ""

    try:
        if article_details and len(article_details) > 0:
            front = article_details[0].get('front', {})
            article_meta = front.get('article-meta', {})

            # Title
            title_group = article_meta.get('title-group', {})
            title = str(title_group.get('article-title', '')).replace('\n', ' ').strip()

            # Abstract
            abstract_sections = article_meta.get('abstract', [])
            if abstract_sections:
                abs_parts = []
                for section in abstract_sections:
                    if hasattr(section, 'get'):
                        paragraphs = section.get('p', [])
                    else:
                        paragraphs = []
                    for p in paragraphs:
                        abs_parts.append(str(p).replace('\n', ' '))
                abstract = ' '.join(abs_parts)

            # Full text body
            body = article_details[0].get('body', {})
            sections = body.get('sec', [])
            for sec in sections:
                paragraphs = sec.get('p', [])
                for p in paragraphs:
                    clean_text = re.sub(r'<.*?>', '', str(p).replace('\n', ' '))
                    full_text += clean_text + '\n'
    except Exception as e:
        st.warning(f"解析文献文本时出错: {e}")

    return title, abstract, full_text


def parse_tsv_to_dataframe(tsv_text: str) -> 'pd.DataFrame':
    """Parse TSV-formatted text into a DataFrame."""
    import pandas as pd
    from io import StringIO
    try:
        header = "化合物\t类型\t毒副作用\n"
        data = StringIO(header + tsv_text)
        df = pd.read_csv(data, sep='\t')
        return df
    except Exception:
        return pd.DataFrame()


import pandas as pd
import re
from pathlib import Path

def clean_author_from_title(title, author):
    # If author is already a proper name (not 'by' or a date), keep it
    if author and author != 'by' and not re.match(r'^\d{4}-\d{2}-\d{2}$', str(author)):
        return author
    
    # Try to extract from title patterns
    # 1. Title - Author - Extra
    match = re.search(r' - ([\w\.\s]+) -', title)
    if match:
        return match.group(1).strip()
    
    # 2. Title by Author.pdf
    match = re.search(r' by ([\w\.\-\s]+)[\.#]', title, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 3. Author - Title
    match = re.search(r'^([\w\.\-\s\']+)\s*-\s*', title)
    if match:
        potential = match.group(1).strip()
        if len(potential.split()) <= 4 and potential.lower() not in ['the', 'a', 'an', 'javascript', 'system']:
            return potential
            
    # 4. Handle DFW_TV, GoodOldNeon, etc.
    if 'DFW' in title: return 'David Foster Wallace'
    if 'CS Lewis' in title: return 'C.S. Lewis'

    return author

def main():
    path = Path('ai_books_tracking/books_enriched.csv')
    df = pd.read_csv(path)
    
    manual_fixes = {
        "Evidence-based Software Engineering.pdf": "Derek M. Jones",
        "The_knight_s_tale Geoffrey Chaucer.pdf": "Geoffrey Chaucer",
        "john boyd.pdf": "John Boyd",
        "situationalawareness.pdf": "Leopold Aschenbrenner",
        "The Good Research Code Handbook.pdf": "Patrick Mineault",
        "stephen_king_on_writing.pdf": "Stephen King",
        "the_martian_chronicles by ray_bradbury.pdf": "Ray Bradbury",
        "What is Man_ and Other Essays, by Mark Twain.pdf": "Mark Twain",
        "Enemies-Of-Promise by CYRIL CONNOLLY.pdf": "Cyril Connolly",
        "Learning_SQL Alan_Beaulieu-EN.pdf": "Alan Beaulieu",
        "Storytelling with Data_ Let’s Practice!-Cole Nussbaumer Knaflic - Wiley (2019).pdf": "Cole Nussbaumer Knaflic",
        "Hitler’s Uranium Club_ The Secret Recordings at Farm Hall - Jeremy Bernstein.pdf": "Jeremy Bernstein",
        "The Pragmatic Programmer_ From Journeyman to Master-Andrew Hunt, David Thomas - Addison-Wesley Professional (1999).pdf": "Andrew Hunt",
        "Supplying War_ Logistics from Wallenstein to Patton-Martin Van Creveld - (1977).pdf": "Martin Van Creveld",
        "Intro Reinforcement Learning from Human Feedback.pdf": "Kaplan",
        "The Inner Game of Tennis_ The Classic Guide to the Mental Side of Peak Performance-W. Timothy Gallwey - (2015).pdf": "W. Timothy Gallwey",
        "bobby-fischer-teaches-chess.pdf": "Bobby Fischer",
        "Privilege_ The Making of an Adolescent Elite at St. Paul’s School (Princeton Studies in Cultural Sociology)-Shamus Rahman Khan - (2010).pdf": "Shamus Rahman Khan",
        "JavaScript_The_Good_Parts_May_2008.pdf": "Douglas Crockford",
        "System Design Interview_ An Insider’s Guide-Alex Xu - (2020).pdf": "Alex Xu",
        "Fluent Python_ Clear, Concise, and Effective Programming-Luciano Ramalho - O’Reilly Media (2022).pdf": "Luciano Ramalho",
        "Parkinson’s Law, and Other Studies in Administration-C. Northcote Parkinson - (1957).pdf": "C. Northcote Parkinson",
        "Anti-Semite_and_Jew_An_Exploration_of_the_Etiology_of_Hate-Jean-Paul Sartre - (1948).pdf": "Jean-Paul Sartre",
        "Working Backwards_ Insights, Stories, and Secrets from Inside Amazon-Colin Bryar, Bill Carr - St. Martin’s Press (2021).pdf": "Colin Bryar",
        "They're Made Out Of Meta - Astral Codex Ten.pdf": "Scott Alexander",
        "A Philosophy of Software Design - John Ousterhout - (2018).pdf": "John Ousterhout",
        "Hilary Mantel - Wolf Hall-HarperCollins (2010).epub": "Hilary Mantel",
        "Hilary Mantel - Bring Up the Bodies-Henry Holt and Co. (2012).epub": "Hilary Mantel",
        "Option Pricing And Volatility - Advanced Strategies And Techniques-Sheldon Natenberg - (1994).pdf": "Sheldon Natenberg",
    }
    
    for title, author in manual_fixes.items():
        mask = df['title'] == title
        if mask.any():
            df.loc[mask, 'author'] = author

    print(f"\nFinal authors: {df['author'].value_counts().head(10)}")
    
    if 'author_original' in df.columns:
        df = df.drop(columns=['author_original'])
    df.to_csv(path, index=False)
    print("\nSaved fixed authors to books_enriched.csv")

if __name__ == "__main__":
    main()

# Elocator
### This project is new and under active development.

A project to help identify the Elo of a chess player based on a set of moves or games. Can be used to identify cheating, game throwing, or other anomalies in a players game.

Currently, an API exists at https://elocator.fly.dev/ which you can use to return positional complexity given an FEN (board state). I am currently developing a website centered around the idea of Elocator. You can see the code behind the API in the Dockerfile as well as the /api/ folder.

In the immediate future, I have a few goals:
1. Make the complexity model much better (incorporate a larger training dataset, a better NN  structure, e.g. HalfKA)
2. Find a mechanism to turn the complexity score into game evaluations
3. Find a mechanism to turn a series of games into a "tournament score".
4. Find a mechanism to identify outliers beyond some percentile (cheating flag).

Longer term, I view this as an opportunity for the chess community to develop "open source cheating detection", among other things.

If you'd like to contribute, please contact me, submit a PR, open an issue, etc.

## Example Usage:

```javascript
async function fetchComplexityScore(fenStr) {
    const url = 'https://elocator.fly.dev/complexity/';
    const data = JSON.stringify({ fen: fenStr });

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: data,
        });

        if (response.ok) {
            const jsonResponse = await response.json();
            document.getElementById('complexityScore').innerText = `Complexity Score: ${jsonResponse.complexity_score}`;
        } else {
            document.getElementById('complexityScore').innerText = 'Error fetching complexity score';
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('complexityScore').innerText = 'Error fetching complexity score';
    }
}

```




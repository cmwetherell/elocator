function submitFEN() {
    const fenStr = document.getElementById('fenInput').value;
    const boardConfig = {
        draggable: false,
        position: fenStr
    };
    
    Chessboard('board', boardConfig);

    fetchComplexityScore(fenStr);
}

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

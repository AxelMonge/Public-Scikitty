
export async function POST(req) {
    try {
        const { board, goal } = await req.json();
        const response = await fetch(`http://127.0.0.1:8001/retraigo`, {
            method: 'Post',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ board: board, goal: goal }),
        });
        if (!response.ok) throw new Error(`Server Error: ${response.status} ${response.statusText}`);
        const result = await response.json();
        return Response.json(result);
    }
    catch (e) {
        return Response.error(e.message);
    }
}
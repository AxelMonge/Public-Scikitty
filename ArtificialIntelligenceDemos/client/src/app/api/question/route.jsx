
export async function POST(req) {
    try {
        const { file_name, pregunta } = await req.json();
        const response = await fetch(`http://127.0.0.1:8001/consulta`, {
            method: 'Post',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ file_name: `${file_name}.json`, pregunta: pregunta }),
        });
        if (!response.ok) throw new Error(`Server Error: ${response.status} ${response.statusText}`);
        const result = await response.json();
        return Response.json(result);
    }
    catch (e) {
        return Response.error(e.message);
    }
}